# gpflow-monitor
Tools for logging an optimisation procedure, specifically designed to
work well with [GPflow](https://github.com/GPflow/GPflow) models.

A very similar structure should work well for general optimisation, which is
what [opt_tools](https://github.com/markvdw/opt_tools) attempts. This
repo is the continuation specifically for GPflow models.

### Setup
`python setup.py develop`

### Usage
This code works with the new branch of GPflow. See `./notebooks/` for
usage examples.

### Support
This code is actively used by researchers who heavily use GPflow. We
will endeavour to
- make sure the code is well-tested,
- the tests will break whenever a breaking change is made to GPflow.

However we cannot put the same effort into maintaining it as is done for
GPflow, simply because logging is not always the priority for our daily
activities. If something is broken, or if you want to improve something,
we will be happy to consider pull requests.
