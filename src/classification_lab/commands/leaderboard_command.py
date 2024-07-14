from ..leaderboard import leaderboard


def register(commands):
    command = commands.add_parser(
        "leaderboard", help="Rank reports sharing dataset and holdout fingerprints"
    )
    command.add_argument("reports", nargs="+")
    command.add_argument("--metric", default="macro_f1")
    command.add_argument("--output")
    command.set_defaults(handler=lambda args: leaderboard(args.reports, args.metric))
