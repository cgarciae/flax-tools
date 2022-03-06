import typing

import jax.numpy as jnp
from flax_tools import utils
from flax_tools.metrics import metrics_utils
from flax_tools.metrics.metric import Metric
from flax_tools.metrics.metrics_utils import AverageMethod, DataType, MDMCAverageMethod

# WARNING: dont use 'typing as tp' here, it will get confused with Accuracy.tp


@utils.dataclass
class Accuracy(Metric):
    r"""
    Computes Accuracy_:

    .. math::
        \text{Accuracy} = \frac{1}{N}\sum_i^N 1(y_i = \hat{y}_i)

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
    tensor of predictions.

    For multi-class and multi-dimensional multi-class data with probability or logits predictions, the
    parameter ``top_k`` generalizes this metric to a Top-K accuracy metric: for each sample the
    top-K highest probability or logit score items are considered to find the correct label.

    For multi-label and multi-dimensional multi-class inputs, this metric computes the "glob"
    accuracy by default, which counts all target or sub-samples separately. This can be
    changed to subset accuracy (which requires all target or sub-samples in the sample to
    be correctly predicted) by setting ``subset_accuracy=True``.

    Accepts all input utils listed in :ref:`references/modules:input utils`.

    Args:
        num_classes:
            Number of classes. Necessary for ``'macro'``, ``'weighted'`` and ``None`` average methods.
        threshold:
            Threshold for transforming probability or logit predictions to binary (0,1) predictions, in the case
            of binary or multi-label inputs. Default value of 0.5 corresponds to input being probabilities.
        average:
            Defines the reduction that is applied. Should be one of the following:

            - ``'micro'`` [default]: Calculate the metric globally, across all samples and classes.
            - ``'macro'``: Calculate the metric for each class separately, and average the
              metrics across classes (with equal weights for each class).
            - ``'weighted'``: Calculate the metric for each class separately, and average the
              metrics across classes, weighting each class by its support (``tp + fn``).
            - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
              the metric for every class.
            - ``'samples'``: Calculate the metric for each sample, and average the metrics
              across samples (with equal weights for each sample).

            .. note:: What is considered a sample in the multi-dimensional multi-class case
                depends on the value of ``mdmc_average``.

            .. note:: If ``'none'`` and a given class doesn't occur in the `preds` or `target`,
                the value for the class will be ``nan``.

        mdmc_average:
            Defines how averaging is done for multi-dimensional multi-class inputs (on top of the
            ``average`` parameter). Should be one of the following:

            - ``None`` [default]: Should be left unchanged if your data is not multi-dimensional
              multi-class.

            - ``'samplewise'``: In this case, the statistics are computed separately for each
              sample on the ``N`` axis, and then averaged over samples.
              The computation for each sample is done by treating the flattened extra axes ``...``
              (see :ref:`references/modules:input utils`) as the ``N`` dimension within the sample,
              and computing the metric for the sample based on that.

            - ``'global'``: In this case the ``N`` and ``...`` dimensions of the inputs
              (see :ref:`references/modules:input utils`)
              are flattened into a new ``N_X`` sample axis, i.e. the inputs are treated as if they
              were ``(N_X, C)``. From here on the ``average`` parameter applies as usual.

        ignore_index:
            Integer specifying a target class to ignore. If given, this class index does not contribute
            to the returned score, regardless of reduction method. If an index is ignored, and ``average=None``
            or ``'none'``, the score for the ignored class will be returned as ``nan``.

        top_k:
            Number of highest probability or logit score predictions considered to find the correct label,
            relevant only for (multi-dimensional) multi-class inputs. The
            default value (``None``) will be interpreted as 1 for these inputs.

            Should be left at default (``None``) for all other utils of inputs.

        multiclass:
            Used only in certain special cases, where you want to treat inputs as a different type
            than what they appear to be. See the parameter's
            :ref:`documentation section <references/modules:using the multiclass parameter>`
            for a more detailed explanation and examples.

        subset_accuracy:
            Whether to compute subset accuracy for multi-label and multi-dimensional
            multi-class inputs (has no effect for other input utils).

            - For multi-label inputs, if the parameter is set to ``True``, then all target for
              each sample must be correctly predicted for the sample to count as correct. If it
              is set to ``False``, then all target are counted separately - this is equivalent to
              flattening inputs beforehand (i.e. ``preds = preds.flatten()`` and same for ``target``).

            - For multi-dimensional multi-class inputs, if the parameter is set to ``True``, then all
              sub-sample (on the extra axis) must be correct for the sample to be counted as correct.
              If it is set to ``False``, then all sub-samples are counter separately - this is equivalent,
              in the case of label predictions, to flattening the inputs beforehand (i.e.
              ``preds = preds.flatten()`` and same for ``target``). Note that the ``top_k`` parameter
              still applies in both cases, if set.

        compute_on_step:
            Forward only calls ``update()`` and return ``None`` if this is set to ``False``.
        dist_sync_on_step:
            Synchronize metric state across processes at each ``forward()``
            before returning the value at the step
        process_group:
            Specify the process group on which synchronization is called.
            default: ``None`` (which selects the entire world)
        dist_sync_fn:
            Callback that performs the allgather operation on the metric state. When ``None``, DDP
            will be used to perform the allgather

    Raises:
        ValueError:
            If ``top_k`` is not an ``integer`` larger than ``0``.
        ValueError:
            If ``average`` is none of ``"micro"``, ``"macro"``, ``"weighted"``, ``"samples"``, ``"none"``, ``None``.
        ValueError:
            If two different input modes are provided, eg. using ``multi-label`` with ``multi-class``.
        ValueError:
            If ``top_k`` parameter is set for ``multi-label`` inputs.

    Example:
        >>> import torch
        >>> from torchmetrics import Accuracy
        >>> target = torch.tensor([0, 1, 2, 3])
        >>> preds = torch.tensor([0, 2, 1, 3])
        >>> accuracy = Accuracy()
        >>> accuracy(preds, target)
        tensor(0.5000)

        >>> target = torch.tensor([0, 1, 2])
        >>> preds = torch.tensor([[0.1, 0.9, 0], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3]])
        >>> accuracy = Accuracy(top_k=2)
        >>> accuracy(preds, target)
        tensor(0.6667)

    """
    tp: typing.Optional[jnp.ndarray] = utils.node()
    fp: typing.Optional[jnp.ndarray] = utils.node()
    tn: typing.Optional[jnp.ndarray] = utils.node()
    fn: typing.Optional[jnp.ndarray] = utils.node()

    threshold: float = utils.static()
    num_classes: typing.Optional[int] = utils.static()
    average: typing.Union[str, AverageMethod] = utils.static()
    mdmc_average: typing.Union[str, MDMCAverageMethod] = utils.static()
    ignore_index: typing.Optional[int] = utils.static()
    top_k: typing.Optional[int] = utils.static()
    multiclass: typing.Optional[bool] = utils.static()
    subset_accuracy: bool = utils.static()
    mode: DataType = utils.static()
    zeros_shape: typing.Tuple[int, ...] = utils.static()

    @classmethod
    def new(
        cls,
        threshold: float = 0.5,
        num_classes: typing.Optional[int] = None,
        average: typing.Union[str, AverageMethod] = AverageMethod.MICRO,
        mdmc_average: typing.Union[str, MDMCAverageMethod] = MDMCAverageMethod.GLOBAL,
        ignore_index: typing.Optional[int] = None,
        top_k: typing.Optional[int] = None,
        multiclass: typing.Optional[bool] = None,
        subset_accuracy: bool = False,
        mode: DataType = DataType.MULTICLASS,
        on: typing.Optional[utils.IndexLike] = None,
        name: typing.Optional[str] = None,
    ):

        if isinstance(average, str):
            average = AverageMethod[average.upper()]

        if isinstance(mdmc_average, str):
            mdmc_average = MDMCAverageMethod[mdmc_average.upper()]

        average = (
            AverageMethod.MACRO
            if average in [AverageMethod.WEIGHTED, AverageMethod.NONE]
            else average
        )

        if average not in [
            AverageMethod.MICRO,
            AverageMethod.MACRO,
            # AverageMethod.SAMPLES,
        ]:
            raise ValueError(f"The `reduce` {average} is not valid.")

        if average == AverageMethod.MACRO and (not num_classes or num_classes < 1):
            raise ValueError(
                "When you set `reduce` as 'macro', you have to provide the number of classes."
            )

        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError(
                f"The `top_k` should be an integer larger than 0, got {top_k}"
            )

        if (
            num_classes
            and ignore_index is not None
            and (not 0 <= ignore_index < num_classes or num_classes == 1)
        ):
            raise ValueError(
                f"The `ignore_index` {ignore_index} is not valid for inputs with {num_classes} classes"
            )

        # Update states
        if average == AverageMethod.SAMPLES:
            raise ValueError(f"The `average` method '{average}' is not yet supported.")

        if mdmc_average == MDMCAverageMethod.SAMPLEWISE:
            raise ValueError(
                f"The `mdmc_average` method '{mdmc_average}' is not yet supported."
            )

        if average == AverageMethod.MICRO:
            zeros_shape = ()
        elif average == AverageMethod.MACRO:
            zeros_shape = (num_classes,)
        else:
            raise ValueError(f'Wrong reduce="{average}"')

        return super().new(
            name=name,
            on=on,
            kwargs=dict(
                tp=None,
                fp=None,
                tn=None,
                fn=None,
                threshold=threshold,
                num_classes=num_classes,
                average=average,
                mdmc_average=mdmc_average,
                ignore_index=ignore_index,
                top_k=top_k,
                multiclass=multiclass,
                subset_accuracy=subset_accuracy,
                mode=mode,
                zeros_shape=zeros_shape,
            ),
        )

    def reset(self) -> "Accuracy":

        initial_value = jnp.zeros(self.zeros_shape, dtype=jnp.uint32)

        return self.replace(
            tp=initial_value,
            fp=initial_value,
            tn=initial_value,
            fn=initial_value,
        )

    def update(self, preds: jnp.ndarray, target: jnp.ndarray, **_) -> "Accuracy":
        """Update state with predictions and targets. See
        :ref:`references/modules:input utils` for more information on input
        utils.

        Args:
            preds: Predictions from module (logits, probabilities, or target)
            target: Ground truth target
        """

        tp, fp, tn, fn = metrics_utils._stat_scores_update(
            preds,
            target,
            intended_mode=self.mode,
            average_method=self.average,
            mdmc_average_method=self.mdmc_average,
            threshold=self.threshold,
            num_classes=self.num_classes,
            top_k=self.top_k,
            multiclass=self.multiclass,
        )

        return self.replace(
            tp=self.tp + tp,
            fp=self.fp + fp,
            tn=self.tn + tn,
            fn=self.fn + fn,
        )

    def compute(self) -> jnp.ndarray:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        # if self.mode is None:
        #     raise RuntimeError("You have to have determined mode.")

        return metrics_utils._accuracy_compute(
            self.tp,
            self.fp,
            self.tn,
            self.fn,
            self.average,
            self.mdmc_average,
            self.mode,
        )
