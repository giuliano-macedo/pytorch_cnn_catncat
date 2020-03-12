import argparse
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
	parser.add_argument("-g"
		"--gpu",
		action="store_true",
		help="use gpu (default: false)",
		dest="gpu"
	)
	parser.add_argument("-F"
		"--feature-extraction",
		action="store_false",
		help="use gpu (default: true)",
		dest="feature_extraction"
	)
	parser.add_argument("-e"
		"--no-epochs",
		type=int,
		help="number of epochs (default: 1000)",
		default=1000,
		dest="no_epochs"
	)

	args=parser.parse_args()

	return args
	