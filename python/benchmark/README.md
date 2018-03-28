# Benchmarking

You can benchmark the speed of:

* Functions
* Solvers
* Graphs

Only a function benchmarking tool is provided so far.

You can execute function benchmarks at this directory.

```shell
pytest
```

This will execute default CPU benchmarks. It take more than 10 minutes to be
done, and outputs benchmark results to the `benchmark-output` folder.

You can specify an extension of special implementations to be executed.
The cudnn extension benchmarks can be run by (in device 0):


```shell
pytest --nnabla-ext=cudnn --nnabla-ext-device-id=0
```

The benchmark results will be placed to the same location as CPU but with other
names accordingly associated with the ext options.


