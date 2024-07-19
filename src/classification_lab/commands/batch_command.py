from ..batch import run_batch


def register(commands):
    command = commands.add_parser("batch", help="Train multiple offline project datasets")
    command.add_argument("datasets", nargs="*")
    command.add_argument("--output-dir", default="reports/batch")
    command.add_argument("--model", choices=["logistic", "forest", "dummy"], default="logistic")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--data-path", default="patient_data.csv")
    command.set_defaults(
        handler=lambda args: run_batch(
            args.datasets or None, args.output_dir, args.model, args.seed, args.data_path
        )
    )
