# Flop Counter for PyTorch Models

fvcore contains a flop-counting tool for pytorch models -- the __first__ tool that can provide both __operator-level__ and __module-level__ flop counts together. We also provide functions to display the results according to the module hierarchy. We hope this tool can help pytorch users analyze their models more easily!

## Existing Approaches:

To our knowledge, a good flop counter for pytorch models that satisfy our needs do not yet exist. We review some existing solutions below:

### Count per-module flops in module-hooks

There are many existing tools (in [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter), [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch), [mmcv](https://github.com/open-mmlab/mmcv/blob/master/mmcv/cnn/utils/flops_counter.py), [pytorch_model_summary](https://github.com/ceykmc/pytorch_model_summary), and our own [mobile_cv](https://github.com/facebookresearch/mobile-vision/blob/master/mobile_cv/lut/lib/pt/flops_utils.py)) that count per-module flops using Pytorch’s [module forward hooks](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=module%20hook#torch.nn.Module.register_forward_hook). They work well for many models, but suffer from the same limitation that makes it hard to get accurate results:

* They are accurate only if every custom module implements a corresponding flop counter. This is too much extra effort for users.
* In addition, determining the flop counter of a complicated module requires manual inspection of its forward code to see what raw ops are called. A refactor of the forward logic   (replacing raw ops by submodules) might require a change in its flop counter.
* When a module contains control flow internally, the counting code has to replicate some of the module’s forward logic.

### Count per-operator flops

These limitations of per-module counting suggest that counting flops at operator level would be better: unlike a large set of custom modules users typically create, custom operators are much less common. Also, operators typically don’t contain control logic that needs to be replicated in its flop counter. For accurate results, operator-level counting is a more preferable approach.

Pytorch’s profiler recently added [flop counting capability](https://github.com/pytorch/pytorch/pull/46506), which accumulates flops for all operators encountered during profiling. However, some features that are highly desirable for research are yet to be supported:

* Module-level aggregation: `nn.Module` is the level of abstraction where users design models. To help design efficient models, providing flops per `nn.Module` in a recursive hierarchy is needed.
* Customization: flops is in fact sometimes ambiguously defined due to research/production needs, or community convention. We’d like the ability to customize the counting by supplying formula for each operator.

### Count *actual* hardware instructions
`perf stat` can collect actual instruction count of a command. After taking into consideration the SIMD instructions, it may be used to compute *actual* total flops that's hardware and implementation dependent. We have also noticed a [blog post](http://www.bnikolic.co.uk/blog/python/flops/2019/10/01/pytorch-count-flops.html) that uses PAPI on intel CPUs to count flops, but this tool can significantly undercount by a factor of 3~4x due to SIMD instructions.

## Our Work

We create a flop counting tool in fvcore, which:

* is accurate for a majority of use cases: it observes all operator calls and collects operator-level flop counts
* can provide aggregated flop counts for each module, and display the flop counts in a hierarchical way
* can be customized from Python to supply flop counting formulas for each operator

It has an interface like this:
```
$ from fvcore.nn import FlopCountAnalysis
$ flops = FlopCountAnalysis(model, input)
$ flops.total()
274656
$ flops.by_operator()
Counter({'conv': 194616, 'addmm': 80040})
$ flops.by_module()
Counter({'': 274656, 'conv1': 48600,
         'conv2': 146016, 'fc1': 69120,
         'fc2': 10080, 'fc3': 840})
$ flops.by_module_and_operator()
{'': Counter({'conv': 194616, 'addmm': 80040}),
 'conv1': Counter({'conv': 48600}),
 'conv2': Counter({'conv': 146016}),
 'fc1': Counter({'addmm': 69120}),
 'fc2': Counter({'addmm': 10080}),
 'fc3': Counter({'addmm': 840})}
```

In addition to providing the results above, the class also allows users to add/override the formula to handle certain ops or ignore certain ops. See [API documentation](https://detectron2.readthedocs.io/en/latest/modules/fvcore.html#fvcore.nn.FlopCountAnalysis) for details.

We further supply functions to pretty-print the results in two styles demonstrated in the image below:
<div align="center" width="600">
  <img src="https://user-images.githubusercontent.com/1381301/116491037-c4a49280-a84d-11eb-8aff-aa5560a8780b.png"/>
</div>

Toy examples are not enough. Below are the pretty-print results of 3 real-world models: we hope they are complicated enough to convince you that the tool probably works for your model as well.

* [Mask R-CNN in detectron2](https://gist.github.com/ppwwyyxx/1885ec8aaf5093a8d40cdde2b6559ab3#file-mask-r-cnn-from-detectron2)
* [Roberta in fairseq](https://gist.github.com/ppwwyyxx/1885ec8aaf5093a8d40cdde2b6559ab3#file-roberta-from-fairseq)
* [ViT in classyvision](https://gist.github.com/ppwwyyxx/1885ec8aaf5093a8d40cdde2b6559ab3#file-vit-from-classyvision)

In addition, our approach is not limited to flop counting, but can collect other operator-level statistics during the execution of a model. For example, [recent research](https://arxiv.org/abs/2003.13678) shows that flop count is poorly correlated with GPU latency, and proposes to use “activation counts” or memory footprint as another metric. We have added `fvcore.nn.ActivationCountAnalysis` that is able to produce this metric as well.


## Appendix: Mechanism & Limitations

Here is briefly how the tool works:

1. It uses pytorch to trace the execution of the model and obtain a graph. This graph records the input/output shapes of every operator, which allows us to compute flops.
2. During tracing, module forward hooks insert per-module information into the graph: upon entering and exiting a module’s forward method, we push/pop the jit tracing scope. After tracing, we use the scopes associated with each operator to figure out which module it belongs to.

The approach still has the following limitations in corner cases, but we think none of them is going to be a deal-breaker for most users (as demonstrated by a few representative models shown above):

1. It `torch.jit.trace` the given model & inputs, which means (1) only `model.forward` is used, but not other methods (2) inputs/outputs of `model.forward` shall be (tuple of) tensors but not arbitrary classes.

   When the above tracing requirements do not satisfy, a simple wrapper around the model is sufficient to make it traceable. (In detectron2 we built a [universal wrapper](https://github.com/facebookresearch/detectron2/blob/543fd075e146261c2e2b0770c9b537314bdae572/detectron2/utils/analysis.py#L63-L65) that recognizes common data structures, to automatically make a model traceable).

2. Forward hooks are only triggered if a module is called by `__call__()`. When a submodule is called with an explicit `.forward()` or other methods, operators may unnaturally contribute to parent modules instead. This doesn’t affect accuracy of total flop counts though.
3. JIT tracing currently prunes away ops that are not used by results. However, as tracing does not capture control flow, it may prune away useful ops whose results only affect control flow. This may lead to under-counting.

   We’d like to see if there are ways to disable the pruning. Meanwhile, it should be very rare that a heavy computation only affects control flow but not directly connected to the final outputs in the computation graph, so this corner case is probably unimportant.
