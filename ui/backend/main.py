from dataclasses import dataclass
import enum
import io
import queue
import re
import tempfile
import threading
from typing import List, Optional
import subprocess

import datetime
import os

import fastapi
import starlette.responses
import pydantic
import hashids
import sqlalchemy as sql
from sqlalchemy import orm
import matplotlib.pyplot as plt


app = fastapi.FastAPI()
hashids = hashids.Hashids(min_length=8, salt=os.environ["PPL_HASHIDS_SALT"])


# https://docs.sqlalchemy.org/en/14/tutorial/engine.html#establishing-connectivity-the-engine
db_url = os.environ["PPL_DATABASE_URL"]
db_kwds = dict()
if db_url.endswith(":memory:"):
    db_kwds.update(
        echo=True,
        # https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#using-a-memory-database-in-multiple-threads
        connect_args={"check_same_thread": False},
        poolclass=sql.pool.StaticPool,
    )
db_engine = sql.create_engine(db_url, **db_kwds, future=True)

Base = orm.declarative_base()

class Computation(Base):
    class Status(enum.Enum):
        queued = "queued"
        in_progress = "in progress"
        success = "success"
        interrupted = "interrupted"
        failed = "failed"

    __tablename__ = "computations"

    id = sql.Column(sql.Integer, primary_key=True)
    submitted_at = sql.Column(sql.DateTime, nullable=False)
    submitted_by = sql.Column(sql.String, nullable=False)
    description = sql.Column(sql.String, nullable=True)
    started_at = sql.Column(sql.DateTime, nullable=True)
    ended_at = sql.Column(sql.DateTime, nullable=True)
    status = sql.Column(sql.Enum(Status), nullable=False)
    failure_reason = sql.Column(sql.String, nullable=True)

    kind = sql.Column(sql.String(50), nullable=False)
    __mapper_args__ = {
        "polymorphic_on": kind,
    }

class ComputationInterrupted(Exception):
    pass

class MrSortModelReconstruction(Computation):
    class Processor(enum.Enum):
        cpu = "CPU"
        gpu = "GPU"

    class WeightsOptimizationStrategy(enum.Enum):
        glop = "glop"
        glop_reuse = "glop-reuse"

    class ProfilesImprovementStrategy(enum.Enum):
        heuristic = "heuristic"
        heuristic_midpoint = "heuristic-midpoints"

    __tablename__ = "mrsort-reconstructions"
    id = sql.Column(sql.Integer, sql.ForeignKey("computations.id"), primary_key=True)
    original_model = sql.Column(sql.String(), nullable=False)
    learning_set_size = sql.Column(sql.Integer, nullable=False)
    learning_set_seed = sql.Column(sql.Integer, nullable=False)
    target_accuracy_percent = sql.Column(sql.Float, nullable=False)
    max_duration_seconds = sql.Column(sql.Integer, nullable=True)
    max_iterations = sql.Column(sql.Integer, nullable=True)
    processor = sql.Column(sql.Enum(Processor), nullable=False)
    seed = sql.Column(sql.Integer, nullable=False)
    weights_optimization_strategy = sql.Column(sql.Enum(WeightsOptimizationStrategy), nullable=False)
    profiles_improvement_strategy = sql.Column(sql.Enum(ProfilesImprovementStrategy), nullable=False)
    reconstructed_model = sql.Column(sql.String, nullable=True)
    accuracy_reached_percent = sql.Column(sql.Float, nullable=True)

    __mapper_args__ = {
        "polymorphic_identity": "mrsort-reconstruction",
    }

    def execute(self):
        with tempfile.TemporaryDirectory() as d:
            original_model_file_name = os.path.join(d, "original-model.txt")
            with open(original_model_file_name, "wt") as f:
                f.write(self.original_model)
            learning_set_file_name = os.path.join(d, "learning-set.txt")
            with open(learning_set_file_name, "wt") as learning_set_file:
                command = [os.path.join(os.getcwd(), "bin/generate-learning-set"), original_model_file_name, str(self.learning_set_size), str(self.learning_set_seed)]
                print("Running", command, flush=True)
                subprocess.run(command, stdout=learning_set_file, check=True, cwd=d)
            command = [
                os.path.join(os.getcwd(), "bin/learn"),
                "--target-accuracy", str(self.target_accuracy_percent),
                "--random-seed", str(self.seed),
                "--force-gpu" if self.processor == MrSortModelReconstruction.Processor.gpu else "--forbid-gpu",
                "--weights-optimization-strategy", self.weights_optimization_strategy.value,
                "--profiles-improvement-strategy", self.profiles_improvement_strategy.value,
                learning_set_file_name,
            ] + (
                [] if self.max_duration_seconds is None else ["--max-duration-seconds", str(self.max_duration_seconds)]
            ) + (
                [] if self.max_iterations is None else ["--max-iterations", str(self.max_iterations)]
            )
            print("Running", command, flush=True)
            process = subprocess.run(command, check=False, capture_output=True, universal_newlines=True, cwd=d)
            if process.returncode <= 1:
                self.reconstructed_model = process.stdout
                if process.returncode == 0:
                    self.accuracy_reached_percent = self.target_accuracy_percent
                else:
                    # @todo Add an option to the tools to print their output in parsable format, including the accuracy reached
                    accuracy_line = process.stderr.splitlines()[-1].strip()
                    m = re.fullmatch(r"Accuracy reached \((.*)%\) is below target", accuracy_line)
                    if m:
                        self.accuracy_reached_percent = float(m.group(1))
                    raise ComputationInterrupted()
            else:
                print(process.stderr)
                process.check_returncode()

Base.metadata.create_all(db_engine)


# An in-process queue is enough for now.
# https://python-rq.org/ will be a suitable alternative for more advanced cases
jobs_queue = queue.Queue()

def dequeue():
    while True:
        id = jobs_queue.get()
        with orm.Session(db_engine) as session:
            computation = session.get(Computation, id)
            computation.started_at = datetime.datetime.now()
            computation.status = Computation.Status.in_progress
            session.commit()

            try:
                computation.execute()
            except ComputationInterrupted:
                computation.status = Computation.Status.interrupted
            except Exception as e:
                computation.status = Computation.Status.failed
                computation.failure_reason = str(e)
            else:
                computation.status = Computation.Status.success
            computation.ended_at = datetime.datetime.now()
            session.commit()
        jobs_queue.task_done()

dequeuing_thread = threading.Thread(target=dequeue, daemon=True)
dequeuing_thread.start()


@dataclass
class MrSortModel:
    criteria_count: int
    categories_count: int
    weights: List[int]
    threshold: float
    profiles: List[List[float]]

    class ParsingError(Exception):
        pass

    @classmethod
    def parse(cls, s):
        parts = " ".join(s.splitlines()).split()  # Ignore actual lines, focus on a list of numbers
        try:
            criteria_count = int(parts[0])
            categories_count = int(parts[1])
            # Don't uses slices in this function (e.g. `parts[2:2 + criteria_count]`) to make sure
            # we get the IndexError if `parts`` is too short
            weights = [float(parts[2 + criterion_index]) for criterion_index in range(criteria_count)]
            threshold = float(parts[2 + criteria_count])
            profiles = [
                [
                    float(parts[3 + (1 + category_index) * criteria_count + criterion_index])
                    for criterion_index in range(criteria_count)
                ]
                for category_index in range(categories_count - 1)
            ]
            return cls(criteria_count, categories_count, weights, threshold, profiles)
        except (IndexError, ValueError):
            raise cls.ParsingError()

@app.get("/mrsort-graph")
def make_mrsort_graph(model: str):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")

    try:
        model = MrSortModel.parse(model)
    except MrSortModel.ParsingError:
        raise fastapi.HTTPException(status_code=422, detail="Model format error")

    xs = list(f"c{criterion_index}\nw={model.weights[criterion_index]}" for criterion_index in range(model.criteria_count))
    for category_index in range(model.categories_count - 1):
        ax.plot(xs, model.profiles[category_index], "o--", label=f"p{category_index}")

    ax.set_ylim(bottom=0, top=1)
    ax.grid(True)
    ax.legend(title=f"λ={model.threshold}")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)

    buf.seek(0)
    return starlette.responses.StreamingResponse(buf, media_type="image/png")


class SubmitMrSortReconstructionInput(pydantic.BaseModel):
    submitted_by: str
    description: Optional[str]
    original_model: str
    learning_set_size: int
    learning_set_seed: int
    target_accuracy_percent: float
    max_duration_seconds: Optional[int]
    max_iterations: Optional[int]
    processor: MrSortModelReconstruction.Processor
    seed: int
    weights_optimization_strategy: MrSortModelReconstruction.WeightsOptimizationStrategy
    profiles_improvement_strategy: MrSortModelReconstruction.ProfilesImprovementStrategy

@app.post("/mrsort-reconstructions")
def submit_mrsort_reconstruction(input: SubmitMrSortReconstructionInput):
    submitted_at = datetime.datetime.now()
    with orm.Session(db_engine) as session:
        computation = MrSortModelReconstruction(
            submitted_at=submitted_at,
            submitted_by=input.submitted_by,
            description=input.description,
            status=Computation.Status.queued,
            original_model=input.original_model,
            learning_set_size=input.learning_set_size,
            learning_set_seed=input.learning_set_seed,
            target_accuracy_percent=input.target_accuracy_percent,
            max_duration_seconds=input.max_duration_seconds,
            max_iterations=input.max_iterations,
            processor=input.processor,
            seed=input.seed,
            weights_optimization_strategy=input.weights_optimization_strategy,
            profiles_improvement_strategy=input.profiles_improvement_strategy,
        )
        session.add(computation)
        session.commit()
        jobs_queue.put(computation.id)
        return computation_of_db(computation)


@app.get("/computations")
def get_computations():
    with orm.Session(db_engine) as session:
        return [computation_of_db(c) for c in session.execute(sql.select(Computation)).scalars()]


@app.get("/computations/{id}")
def get_computation(id: str):
    id = hashids.decode(id)
    if len(id) == 1:
        id = id[0]
        with orm.Session(db_engine) as session:
            computation = session.get(Computation, id)
            if computation is not None:
                return computation_of_db(computation)
    raise fastapi.HTTPException(status_code=404, detail="Computation not found")


def computation_of_db(computation: Computation):
    duration_seconds = None
    if computation.ended_at is not None:
        duration_seconds = (computation.ended_at - computation.started_at).total_seconds()
    c = dict(
        computation_id=hashids.encode(computation.id),
        submitted_at=computation.submitted_at.strftime("%Y-%m-%d %H:%M:%S"),
        submitted_by=computation.submitted_by,
        description=computation.description,
        kind=computation.kind,
        status=computation.status,
        failure_reason=computation.failure_reason,
        duration_seconds=duration_seconds,
    )
    if computation.kind == "mrsort-reconstruction":
        c.update(
            original_model=computation.original_model,
            learning_set_size=computation.learning_set_size,
            learning_set_seed=computation.learning_set_seed,
            target_accuracy_percent=computation.target_accuracy_percent,
            max_duration_seconds=computation.max_duration_seconds,
            max_iterations=computation.max_iterations,
            processor=computation.processor,
            seed=computation.seed,
            weights_optimization_strategy=computation.weights_optimization_strategy,
            profiles_improvement_strategy=computation.profiles_improvement_strategy,
            reconstructed_model=computation.reconstructed_model,
            accuracy_reached_percent=computation.accuracy_reached_percent,
        )
    else:
        assert False
    return c
