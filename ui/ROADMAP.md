This file describes the first iteration of a web-based GUI, for validation by Laurent.

The main goal of this GUI is to make demos easier, and let people play with the underlying tools without having to install them.

Main use case: reconstitute a model
===================================

The user has an MR-sort model, and wants to:

- generate a pseudo-random learning set from this model
- learn a model from the learning set
- compare the reconstituted model with the initial one

Mock-ups
========

Landing page
------------

This page is at the root of the web application (path `/`).

<!-- Mockups in this document have been created using https://github.com/evolus/pencil. The primary file is mockups.epgz, and *.png files are exported using menu "Export..." -->
![Landing page mockup](landing.png)

### "New computation" section

- the "MR-Sort model reconstruction" link takes the user to the "Submission" page described below
- there is only one kind of computation for now, but I expect this to change soon, so I already design the UI to allow it

### "Existing computations" section

- the "Description" column is a free text set when submitting the computation
- the "Status" column can have the following values:
    - "queued": the computation has been submitted but has not started yet
    - "in progress": the computation has started
    - "success": the computation has finished and reached its goal
    - "interrupted": the computation took longer than allowed, but did not error
    - "failed": the computation failed. The cause of the error is displayed
- in the "Results" column, the link takes the user to the appropriate "Results" page described below
- the table is sorted by "Submitted at"
- there is no custom sorting
- there is no filtering

Submission page
---------------

This page is at path `/submit/mrsort-reconstruction`.

![Submission page mockup](mrsort_model_reconstruction_submission.png)

### Preamble

- the "Submitted by" field is stored in a cookie to avoid having to type it every time
- the "Description" column is a free text used to attach any information to the computation (context, details, references, etc.)

### "Original model" section

- the model can be loaded from a file and/or edited in place using the syntax implemented by `ppl::io::Model`. I'll document that syntax properly during the development of this GUI
- the graph is updated live when the model is changed

### "Learning set generation" section

- the "Pseudo-random seed" field is initialized randomly by the GUI but can be overridden by the user

### "Model reconstruction" section

#### "Termination criteria" sub-section

- this section list all the criteria that can terminate the computation, similarly to the command-line of the `learn` tool
- computation is terminated as soon as the first criterion is met
- computation reaches state "success" if that criterion is the "Target accuracy", and state "interrupted" otherwise

#### "Algorithm" sub-section

- in this section, the user can tweak all aspects of the algorithm used
- it will evolve as more policies are added to the project

### Footer

- the "Submit" button redirects to the "Results" page described below

Results page
------------

This page is at path `/computations/<id>` where `<id>` is an opaque identifier made of 8 letters and digits.

![Results page mockup](mrsort_model_reconstruction_results.png)

- for a queued or in progress computation, the following message is added to the "Status" field: "This page automatically refreshes every 10 seconds. You can also come back later."
- for a failed computation the error message is added to the "Status" field

Technical choices
=================

This GUI is web-based. The server part runs in Docker, using `docker-compose`. The client part is usable in Firefox and Chrome.

The dynamic part of the server is written in Python.
SQLite3 is used to persist data, through SQLAlchemy.

Note: persisting with SQLite forbids serving the application from several machines.
This forbids scaling up the application.

The presentation layer uses the Bootstrap CSS framework, to provide a responsive appearance. (Responsive: that adapts to any size of screen)

We had to choose between two architectures with different tradeoffs detailed below, and we chose option (A):

- (A) The dynamic part of the server is a REST API
- (B) The dynamic part of the server generates HTML

## (A) The dynamic part of the server is a REST API

In that option, the client is a Vue.js application served statically.
It communicates with the dynamic part of the server through a REST API.
The dynamic part of the server uses FastAPI (or maybe Flask).
A NGinx revers proxy is used to unify serving the static client and the dynamic back-end.

Note that the NGinx reverse proxy is a perfect place to implement a (crude) password protection for the application.

Advantages:

- good separation of concerns between the actual GUI and the "glue" layer between the GUI and the tools
- Vue.js make it easy to build reactive applications. For example, reactivity makes it easy to synchronize the model and its graph in mockups above. Reactive applications are typically more attractive to their end-user, which could help with the "wahou" effect
- I've already created several applications this way so I know where I'm going

## (B) The dynamic part of the server generates HTML

In that option, the client is made of HTML served directly by the dynamic part of the server.
A few dynamic parts can be added in the client using jQuery.
Each action from the user loads a new page.
The server is written using Flask.

Advantages:

- more traditional
- may be easier to understand and maintain by people unfamiliar with reactive technologies
- relative simplicity: a reverse proxy is not required so there are fewer components

## Personal recommendation

I recommend choosing option (A).
I think the initial development will be barely longer, and I'm sure the maintenance (by myself at least) will be much easier.
