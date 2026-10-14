"""CLI commands for managing AtomWorks PyMOL editor extensions."""
# ruff: noqa: B008

import typer

app = typer.Typer(help="Manage AtomWorks PyMOL editor extensions.")


@app.command("add")
def add_extension(
    module: str = typer.Argument(..., help="Dotted module path, e.g. proteinfoundation.pymol.editors"),
) -> None:
    """Register a PyMOL editor extension module.

    The module must be importable and must expose a ``register()`` function.
    The registration takes effect on the next PyMOL launch (or immediately
    via Plugin → AtomWorks → Extensions if PyMOL is running).

    Example:
        atomworks pymol add proteinfoundation.pymol.editors
    """
    import importlib

    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        typer.secho(f"Cannot import '{module}': {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from None

    if not callable(getattr(mod, "register", None)):
        typer.secho(f"Module '{module}' has no register() function.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    from atomworks.pymol_plugin.extensions import add_extension as _add

    if _add(module):
        typer.secho(f"Registered '{module}'.", fg=typer.colors.GREEN)
        typer.echo("Takes effect on next PyMOL launch.")
    else:
        typer.secho(f"'{module}' is already registered.", fg=typer.colors.YELLOW)


@app.command("remove")
def remove_extension(
    module: str = typer.Argument(..., help="Dotted module path to remove"),
) -> None:
    """Unregister a PyMOL editor extension module.

    Example:
        atomworks pymol remove proteinfoundation.pymol.editors
    """
    from atomworks.pymol_plugin.extensions import remove_extension as _remove

    if _remove(module):
        typer.secho(f"Removed '{module}'.", fg=typer.colors.GREEN)
    else:
        typer.secho(f"'{module}' was not registered.", fg=typer.colors.YELLOW)


@app.command("list")
def list_extensions() -> None:
    """List all registered PyMOL editor extensions and their import status.

    Example:
        atomworks pymol list
    """
    import importlib

    from atomworks.pymol_plugin.extensions import extensions_file, read_extensions

    modules = read_extensions()
    if not modules:
        typer.echo(f"No extensions registered. (config: {extensions_file()})")
        return

    typer.echo(f"Registered extensions ({extensions_file()}):")
    for module in modules:
        try:
            importlib.import_module(module)
            status = typer.style("✓ importable", fg=typer.colors.GREEN)
        except ImportError:
            status = typer.style("✗ not found", fg=typer.colors.RED)
        typer.echo(f"  {module}  {status}")
