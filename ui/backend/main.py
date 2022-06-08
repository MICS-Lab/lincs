import queue
import threading
from typing import Optional

import datetime
import os
import time  # @todo Remove all calls to time.sleep

import fastapi
import pydantic
import hashids
import sqlalchemy as sql
from sqlalchemy import orm


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

    kind = sql.Column(sql.String(50), nullable=False)
    __mapper_args__ = {
        "polymorphic_on": kind,
    }

class MrSortModelReconstruction(Computation):
    __tablename__ = "mrsort-reconstructions"
    id = sql.Column(sql.Integer, sql.ForeignKey("computations.id"), primary_key=True)
    original_model = sql.Column(sql.String(), nullable=False)
    learning_set_size = sql.Column(sql.Integer, nullable=False)
    learning_set_seed = sql.Column(sql.Integer, nullable=False)
    target_accuracy_percent = sql.Column(sql.Float, nullable=False)
    max_duration_seconds = sql.Column(sql.Float, nullable=True)
    max_iterations = sql.Column(sql.Integer, nullable=True)
    processor = sql.Column(sql.String)  # @todo Use an enum?
    seed = sql.Column(sql.Integer, nullable=False)

    __mapper_args__ = {
        "polymorphic_identity": "mrsort-reconstruction",
    }

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

        time.sleep(15)  # @todo Actually implement

        with orm.Session(db_engine) as session:
            computation = session.get(Computation, id)
            computation.ended_at = datetime.datetime.now()
            computation.status = "success"
            session.commit()
        jobs_queue.task_done()

dequeuing_thread = threading.Thread(target=dequeue, daemon=True)
dequeuing_thread.start()


class SubmitMrSortReconstructionInput(pydantic.BaseModel):
    submitted_by: str
    description: Optional[str]
    original_model: str
    learning_set_size: int
    learning_set_seed: int
    target_accuracy_percent: float
    max_duration_seconds: Optional[float]
    max_iterations: Optional[float]
    processor: str  # @todo Use an enum
    seed: int

@app.post("/mrsort-reconstructions")
def submit_mrsort_reconstruction(input: SubmitMrSortReconstructionInput):
    submitted_at = datetime.datetime.now()
    time.sleep(0.5)
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
    time.sleep(0.5)
    with orm.Session(db_engine) as session:
        return [computation_of_db(c) for c in session.execute(sql.select(Computation)).scalars()]


@app.get("/computations/{id}")
def get_computation(id: str):
    time.sleep(0.5)
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
        )
    else:
        assert False
    return c
