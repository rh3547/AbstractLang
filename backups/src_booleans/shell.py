import abstract
import sys

mode = None
if len(sys.argv) == 2:
    mode = sys.argv[1]

    if mode == "lexer":
        print("[== Testing Lexer ==]")
    elif mode == "parser":
        print("[== Testing Parser ==]")

while True:
    text = input('abstract > ')

    result = None
    error = None

    if mode == "lexer":
        result, error = abstract.test_lexer('<stdin>', text)
    elif mode == "parser":
        result, error = abstract.test_parser('<stdin>', text)
    else:    
        result, error = abstract.run('<stdin>', text)

    if error: print(error.as_string())
    else: print(result)