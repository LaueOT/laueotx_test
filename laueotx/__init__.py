import click
# from laueotx.apps.fakeapp import main as fakeapp
from laueotx.apps.realdata import main as realdata

@click.group()
def main():
    pass

# main.add_command(fakeapp)
main.add_command(realdata)

if __name__ == "__main__":
    main()