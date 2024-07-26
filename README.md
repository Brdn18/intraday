# Intraday modified

This is a fork from original https://github.com/diovisgood/intraday which seems to be abandoned

### Changes
- Introduced new provider - Randomwalk
- Two runners
    - runner.py - single runner which creates environment, trains agent, evaluates returns. Uses randomwalk as data source. Can plot graphs of price walk and return from trained agent
    - runner_many.py - set of runners with different training params and different training algorithms for agents. Allows gathering stats from many runs and estimate average returns


### Drawbacks

Outdated versions of gym and pyglet are used. They can be updated, but it would break everything

Tested on python3.10

