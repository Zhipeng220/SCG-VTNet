"""Run the original SCG-VTNet training and evaluation entry point."""

from scg_vtnet.training.train_core import main
from scg_vtnet.training.core_arguments import get_args


if __name__ == "__main__":
    main(get_args())
