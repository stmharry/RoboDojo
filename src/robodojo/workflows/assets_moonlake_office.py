"""Moonlake office asset-builder command facade."""


def main() -> None:
    from robodojo.workflows.asset_builders.moonlake_office.publication import main as run

    run()


if __name__ == "__main__":
    main()
