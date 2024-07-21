# Problem statement: Write a Python script that takes two numbers and a basic arithmetic operation (add, subtract, multiply, divide) as input and outputs the result.
#
# Going to do this using reverse polish notation.


def calculate(num1, num2, operation):
    """
    Performs basic arithmetic operations on two numbers.

    Parameters:
    - num1 (float): The first number.
    - num2 (float): The second number.
    - operation (str): The operation to perform. Expected values are '+', '-', '*', '/'.

    Returns:
    - float or int: The result of the arithmetic operation. Returns an integer if the result is a whole number.
    """
    if operation == "+":
        result = num1 + num2
    elif operation == "-":
        result = num1 - num2
    elif operation == "*":
        result = num1 * num2
    elif operation == "/":
        if num2 == 0:
            return "Error: Division by zero is not allowed."
        result = num1 / num2
    else:
        return "Invalid operation. Please enter '+', '-', '*', or '/'."

    if result.is_integer():
        return int(result)
    else:
        return result


if __name__ == "__main__":
    ans = None  # Initialize ans variable to store the result of the previous calculation
    while True:
        print(
            "Enter two numbers and an operation (add, subtract, multiply, divide) separated by spaces, or 'exit' to quit: ",
            end="",
        )
        input_str = input()
        if input_str.lower() == "exit":
            break
        input_list = input_str.split()

        try:
            if input_list[0] == "ANS":
                if ans is None:
                    raise ValueError(
                        "Previous result is not available. Please perform a calculation first."
                    )
                else:
                    num1 = ans
            else:
                num1 = float(input_list[0])
            if input_list[1] == "ANS":
                if ans is None:
                    raise ValueError(
                        "Previous result is not available. Please perform a calculation first."
                    )
                num2 = ans
            else:
                num2 = float(input_list[1])
            operation = input_list[2]

            # Check if the user wants to use the previous result
            if (
                num1 == 0
                and num2 == 0
                and operation == "ANS"
                and ans is not None
            ):
                num1 = ans  # Use the previous result as the first number

            result = calculate(num1, num2, operation)
            ans = result  # Store the result for future calculations
            print(f"The result is: {result}")
        except ValueError:
            print("Please enter two numbers followed by an operation.")
        except IndexError:
            print(
                "Input format error. Please enter two numbers followed by an operation."
            )
