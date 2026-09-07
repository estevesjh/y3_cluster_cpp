# `src/modules/`

The tests directly associated with `src/modules` are the integration-template
examples. The production modules are exercised indirectly by the DES Y3 tests
listed in {doc}`src_pipelines_des_y3` (registered centrally in
`test/CMakeLists.txt`).

| CTest target | Test source | Module under test | What it tests | Status |
|---|---|---|---|---|
| `test_ExampleScalarIntegrationModule` | [`src/modules/ExampleScalar/test_ExampleScalarIntegrationModule.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/ExampleScalar/test_ExampleScalarIntegrationModule.cc) | [`ExampleScalarIntegrationModule`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/ExampleScalar/ExampleScalarIntegration_module.cc) | Scalar CosmoSIS integration-module wiring | Passing |
| `ExampleScalarIntegrand_test` | [`ExampleScalarIntegrand.test.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/ExampleScalar/ExampleScalarIntegrand.test.cc) | [`ExampleScalarIntegrand.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/ExampleScalar/ExampleScalarIntegrand.cc) | Example scalar integrand evaluation and quadrature | Passing |
| `test_ExampleVectorIntegrationModule` | [`src/modules/ExampleVector/test_ExampleVectorIntegrationModule.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/ExampleVector/test_ExampleVectorIntegrationModule.cc) | [`ExampleVectorIntegrationModule`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/ExampleVector/ExampleVectorIntegration_module.cc) | Vector CosmoSIS integration-module wiring | Passing |
| `ExampleVectorIntegrand_test` | [`ExampleVectorIntegrand.test.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/ExampleVector/ExampleVectorIntegrand.test.cc) | [`ExampleVectorIntegrand.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/ExampleVector/ExampleVectorIntegrand.cc) | Example vector integrand evaluation and quadrature | Passing |
| `test_ExampleOneDIntegrationModule` | [`src/modules/ExampleOneD/test_ExampleOneDIntegrationModule.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/ExampleOneD/test_ExampleOneDIntegrationModule.cc) | [`ExampleOneDIntegration_module.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/ExampleOneD/ExampleOneDIntegration_module.cc) | One-dimensional CosmoSIS integration-module wiring | Passing |
| `ExampleOneDIntegrand_test` | [`ExampleOneDIntegrand.test.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/ExampleOneD/ExampleOneDIntegrand.test.cc) | [`ExampleOneDIntegrand.cc`](https://github.com/estevesjh/y3_cluster_cpp/blob/master/src/modules/ExampleOneD/ExampleOneDIntegrand.cc) | Example 1D integrand evaluation and quadrature | Passing |
