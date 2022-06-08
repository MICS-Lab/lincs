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
    submitted_by = sql.Column(sql.String)

Base.metadata.create_all(db_engine)


class CreateComputationInput(pydantic.BaseModel):
    submitted_by: str


@app.post("/computations")
def create_computation(computation: CreateComputationInput):
    time.sleep(0.5)
    with orm.Session(db_engine) as session:
        c = Computation(submitted_by=computation.submitted_by)
        session.add(c)
        session.commit()
        return computation_of_db(c)


@app.get("/computations")
def get_computations():
    time.sleep(0.5)
    with orm.Session(db_engine) as session:
        return [computation_of_db(c) for c in session.execute(sql.select(Computation)).scalars()]


@app.get("/computations/{id}")
def get_computation(id: str):
    time.sleep(0.5)
    id = hashids.decode(id)[0]
    with orm.Session(db_engine) as session:
        c = session.get(Computation, id)
        return computation_of_db(c)


def computation_of_db(c: Computation):
    return dict(
        computation_id=hashids.encode(c.id),
        submitted_by=c.submitted_by,
    )
