import argparse
import models
def get_args():
	parser=argparse.ArgumentParser()
	parser.add_argument("dataset",
		type=str,
		default=None,
		help="dataset path (default: download dataset)",
		nargs="?"
	)
	parser.add_argument("-lr",
		type=float,
		default=1e-3,
		help="learning rate (default: 1e-3)"
	)
	parser.add_argument("-m",
		"--model",
		type=str,
		default="V1",
		choices=models.choices,
		help="model to use (default: V1)"
	)
	parser.add_argument("-g"
		"--gpu",
		action="store_true",
		help="use gpu (default: false)",
		dest="gpu"
	)
	parser.add_argument("-e"
		"--no-epochs",
		type=int,
		help="number of epochs (default: 1000)",
		default=1000,
		dest="no_epochs"
	)

	args=parser.parse_args()

	args.model=vars(models)[args.model]()

	return args
	