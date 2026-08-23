from src.data_loader import load_data


def test_missing_file():
    try:
        load_data("tests/does_not_exist.csv")
    except FileNotFoundError as error:
        print("PASS - Missing file handled correctly")
        print(error)


def test_unsupported_file():
    with open("tests/test.txt", "w") as file:
        file.write("This is a test file.")

    try:
        load_data("tests/test.txt")
    except ValueError as error:
        print("PASS - Unsupported file handled correctly")
        print(error)


def test_invalid_path():
    try:
        load_data("tests")
    except ValueError as error:
        print("PASS - Directory path handled correctly")
        print(error)


if __name__ == "__main__":
    test_missing_file()
    test_unsupported_file()
    test_invalid_path()