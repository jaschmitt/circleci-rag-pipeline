
# custom libs
import chains
import sys



def read_input():
    try:
        return input("Enter text: ")
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)



# initialize starter chain
#chain = chains.assistant_chain("Bob").getChain()
chain = chains.documentation_chain("https://docs.smith.langchain.com").getChain()


print("Entering LLM conversation. Hit Ctrl-C to quit...")


# constantly read in input from command line
while True:
    user_input = read_input()

    # Check if the user wants to exit
    if user_input.lower() == "exit":
        print("Exiting...")
        break

    # Pass the input through the language model chain
    result = chain.invoke({"question": user_input})

    # Display the result
    print(result, flush=True)

    # create spacer before next question
    print("")
    print("")

