import simpy

def simpy_run(tasks, es_dict, allocations):
    env = simpy.Environment()
    es_resources = {name: simpy.Resource(env, int(es.capacity)) for name, es in es_dict.items()}
    log = []

    def task_proc(task):
        assigned = allocations.get(task.id, {})
        for es_name, units in assigned.items():
            es_resource = es_resources[es_name]
            with es_resource.request() as req:
                yield req
                yield env.timeout(units)
                log.append((env.now, f"Task {task.id}: {units} units done at {es_name}"))
    for task in tasks:
        env.process(task_proc(task))
    env.run()
    return log
