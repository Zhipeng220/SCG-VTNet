"""Train or evaluate the model comparison and matched ASL variants."""

from scg_vtnet.training.model_comparison import main
from scg_vtnet.training.comparison_arguments import get_args


if __name__ == "__main__":
    main(get_args())
