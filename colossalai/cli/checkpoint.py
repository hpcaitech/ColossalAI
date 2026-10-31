import json
from pathlib import Path

import click


@click.group()
def checkpoint():
    """Checkpoint format tools."""


@checkpoint.command("migrate-optimizer")
@click.argument("source", type=click.Path(exists=True, path_type=Path))
@click.argument("destination", type=click.Path(path_type=Path))
@click.option(
    "--trust-legacy-pickle",
    is_flag=True,
    required=True,
    help="Allow legacy decoding, which can execute code. Use only trusted input in an isolated environment.",
)
def migrate_optimizer(source, destination, trust_legacy_pickle):
    """Convert a trusted legacy optimizer file or directory to data-only v2."""
    from colossalai.checkpoint_io.optimizer_migration import migrate_optimizer_checkpoint

    try:
        result = migrate_optimizer_checkpoint(source, destination, trust_legacy_pickle=trust_legacy_pickle)
        click.echo(json.dumps(result, indent=2))
    except (OSError, ValueError, TypeError) as exc:
        raise click.ClickException(str(exc)) from exc
