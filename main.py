import buildPairs, generation
import trainSiamese


def DisplayMenu():
    print("===== Menu =====")
    print("1. Build Pairs")
    print("2. Train Model")
    print("3. Generate Images")
    print("4. Exit")
    print("=================")

def ExecuteChoice(choice):
    if choice == 1:
        buildPairs.load_pairs()
    elif choice == 2:
        trainSiamese.trainSiamese()
    elif choice == 3:
        path = input("Enter the path of your images: ")
        num = int(input("How many distortions do you want "))
        generation.GenerateDistorted(path, num)


def main():
    choice = 0
    while choice != 4:
        DisplayMenu()
        try:
            choice = int(input("Enter Choice Number: "))
            while choice > 4:
                print("Please Enter A Valid Choice: ", end="")
                choice = int(input())
            ExecuteChoice(choice)
            input("Press Enter to continue...")
        except ValueError as e:
            print("Invalid input. Please enter a number.")
            print(e)


if __name__ == "__main__":
    main()
