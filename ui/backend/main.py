import os
import time  # @todo Remove all calls to time.sleep

import fastapi
import pydantic
import hashids


app = fastapi.FastAPI()
hashids = hashids.Hashids(min_length=8, salt=os.environ.get("PPL_HASHID_SALT", "Default salt"))


computations = []  # @todo Remove, store in SQLite


class CreateComputationInput(pydantic.BaseModel):
    submitted_by: str


@app.post("/computations")
def create_computation(computation: CreateComputationInput):
    id = len(computations)
    c = dict(
        computation_id=hashids.encode(id),
        submitted_by=computation.submitted_by,
    )
    computations.append(c)
    time.sleep(1)
    return c


@app.get("/computations")
def get_computations():
    time.sleep(1)
    return computations


@app.get("/computations/{id}")
def get_computation(id: str):
    time.sleep(1)
    id = hashids.decode(id)[0]
    return computations[id]
