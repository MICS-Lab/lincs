import io
import queue
import re
import tempfile
import threading
from typing import Optional
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
hashids = hashids.Hashids(min_length=8, salt=os.environ.get("PPL_HASHID_SALT", "Default salt"))


# https://docs.sqlalchemy.org/en/14/tutorial/engine.html#establishing-connectivity-the-engine
db_engine = sql.create_engine(
    "sqlite+pysqlite:///:memory:",  # @todo Write to disc (then remove the next lines)
    # https://docs.sqlalchemy.org/en/13/dialects/sqlite.html#using-a-memory-database-in-multiple-threads
    connect_args={"check_same_thread": False},
    poolclass=sql.pool.StaticPool,
    # End of lines to remove
    echo=True,  # @todo Remove
    future=True,
)

Base = orm.declarative_base()

class Computation(Base):
    __tablename__ = "computations"

    id = sql.Column(sql.Integer, primary_key=True)
    submitted_at = sql.Column(sql.DateTime, nullable=False)
    submitted_by = sql.Column(sql.String, nullable=False)
    description = sql.Column(sql.String, nullable=True)
    started_at = sql.Column(sql.DateTime, nullable=True)
    ended_at = sql.Column(sql.DateTime, nullable=True)
    status = sql.Column(sql.String(50), nullable=False)  # @todo Use an enum?
    failure_reason = sql.Column(sql.String, nullable=True)

    kind = sql.Column(sql.String(50), nullable=False)
    __mapper_args__ = {
        "polymorphic_on": kind,
    }

class ComputationInterrupted(Exception):
    pass

class MrSortModelReconstruction(Computation):
    __tablename__ = "mrsort-reconstructions"
    id = sql.Column(sql.Integer, sql.ForeignKey("computations.id"), primary_key=True)
    original_model = sql.Column(sql.String(), nullable=False)
    learning_set_size = sql.Column(sql.Integer, nullable=False)
    learning_set_seed = sql.Column(sql.Integer, nullable=False)
    target_accuracy_percent = sql.Column(sql.Float, nullable=False)
    max_duration_seconds = sql.Column(sql.Integer, nullable=True)
    max_iterations = sql.Column(sql.Integer, nullable=True)
    processor = sql.Column(sql.String)  # @todo Use an enum?
    seed = sql.Column(sql.Integer, nullable=False)
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
                subprocess.run(
                    [os.path.join(os.getcwd(), "bin/generate-learning-set"), original_model_file_name, str(self.learning_set_size), str(self.learning_set_seed)],
                    stdout=learning_set_file,
                    check=True,
                    cwd=d,
                )
            process = subprocess.run(
                [
                    os.path.join(os.getcwd(), "bin/learn"),
                    "--target-accuracy", str(self.target_accuracy_percent),
                    "--random-seed", str(self.seed),
                    "--force-gpu" if self.processor == "GPU" else "--forbid-gpu",
                    learning_set_file_name,
                ] + (
                    [] if self.max_duration_seconds is None else ["--max-duration-seconds", str(self.max_duration_seconds)]
                ) + (
                    [] if self.max_iterations is None else ["--max-iterations", str(self.max_iterations)]
                ),
                check=False,
                capture_output=True, universal_newlines=True,
                cwd=d,
            )
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
            computation.status = "in progress"
            session.commit()

            try:
                computation.execute()
            except ComputationInterrupted:
                computation.status = "interrupted"
            except Exception as e:
                computation.status = "failed"
                computation.failure_reason = str(e)
            else:
                computation.status = "success"
            computation.ended_at = datetime.datetime.now()
            session.commit()
        jobs_queue.task_done()

dequeuing_thread = threading.Thread(target=dequeue, daemon=True)
dequeuing_thread.start()


@app.get("/mrsort-graph")
def make_mrsort_graph(model: str):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")

    model = model.split()
    criteria_count = int(model[0])
    categories_count = int(model[1])
    # weights = model[2:2 + criteria_count]
    # threshold = model[2 + criteria_count]
    xs = list(range(criteria_count))
    for category_index in range(categories_count - 1):
        profile = [float(y) for y in model[3 + (1 + category_index) * criteria_count:3 + (2 + category_index) * criteria_count]]
        ax.plot(xs, profile)

    ax.set_ylim(bottom=0, top=1)
    ax.set_xlim(left=0, right=criteria_count - 1)

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
    processor: str  # @todo Use an enum
    seed: int

@app.post("/mrsort-reconstructions")
def submit_mrsort_reconstruction(input: SubmitMrSortReconstructionInput):
    submitted_at = datetime.datetime.now()
    with orm.Session(db_engine) as session:
        computation = MrSortModelReconstruction(
            submitted_at=submitted_at,
            submitted_by=input.submitted_by,
            description=input.description,
            status="queued",
            original_model=input.original_model,
            learning_set_size=input.learning_set_size,
            learning_set_seed=input.learning_set_seed,
            target_accuracy_percent=input.target_accuracy_percent,
            max_duration_seconds=input.max_duration_seconds,
            max_iterations=input.max_iterations,
            processor=input.processor,
            seed=input.seed,
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
    # @todo Return a 404 when not found
    id = hashids.decode(id)[0]
    with orm.Session(db_engine) as session:
        computation = session.get(Computation, id)
        return computation_of_db(computation)


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
            reconstructed_model=computation.reconstructed_model,
            accuracy_reached_percent=computation.accuracy_reached_percent,
        )
    else:
        assert False
    return c
