/* node: the tests read SM_MODEL_DIR and the other variables from the process environment */
Module.preRun = [() => typeof ENV == "object" && Object.assign(ENV, process.env)];
