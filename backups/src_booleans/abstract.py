#######################################
# IMPORTS
#######################################

from logging.config import IDENTIFIER
from string_with_arrows import *
import string

################################################################
# CONSTANTS
################################################################

DIGITS = '0123456789'

LETTERS = string.ascii_letters

LETTERS_DIGITS = LETTERS + DIGITS

IDENTIFIER_CHARS = LETTERS + DIGITS + '_'

################################################################
# ERRORS
################################################################

class Error:
    def __init__(self, pos_start, pos_end, name, details):
        self.pos_start = pos_start
        self.pos_end = pos_end
        self.name = name
        self.details = details

    def as_string(self):
        result = self.generate_error_header()
        result += f'\n  File "{self.pos_start.filename}", line {self.pos_start.line + 1}, index {self.pos_start.index}'
        result += self.generate_error_preview()
        result += '\n'
        return result

    def generate_error_header(self):
        return f'\n[ERROR] {self.name}: {self.details}'

    def generate_error_preview(self):
        return '\n\nError Preview:\n' + string_with_arrows(self.pos_start.filetext, self.pos_start, self.pos_end)

class IllegalCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Illegal Character', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

class ExpectedCharError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Expected Character', details)

class RuntimeError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end, 'Runtime Error', details)
        self.context = context

    def as_string(self):
        result = super().generate_error_header()
        result += self.generate_traceback()
        result += super().generate_error_preview()
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        context = self.context

        while context:
            result = f'\n  File "{pos.filename}", line {str(pos.line + 1)}, index {pos.index}, in {context.display_name}' + result
            pos = context.parent_entry_pos
            context = context.parent
            
        return '\nTraceback (most recent call last):' + result

class Position:
    def __init__(self, index, line, col, filename, filetext):
        self.index = index
        self.line = line
        self.col = col
        self.filename = filename
        self.filetext = filetext
    
    def advance(self, current_char=None):
        self.index += 1
        self.col += 1

        if current_char == "\n":
            self.line += 1
            self.col = 0
        
        return self
    
    def copy(self):
        return Position(self.index, self.line, self.col, self.filename, self.filetext)

################################################################
# TOKENS
################################################################

# Types
TT_INT              = 'INT'              # Integer

TT_FLOAT            = 'FLOAT'            # Float


# Operators
TT_PLUS             = 'PLUS'             # Plus
TS_PLUS             = '+'                # Plus Symbol

TT_MINUS            = 'MINUS'            # Minus
TS_MINUS            = '-'                # Minus Symbol

TT_MULT             = 'MULT'             # Multiply
TS_MULT             = '*'                # Multiply Symbol

TT_DIV              = 'DIV'              # Divide
TS_DIV              = '/'                # Divide Symbol

TT_POW              = 'POW'              # Power
TS_POW              = '^'                # Power Symbol

TT_LPAREN           = 'LPAREN'           # Left Parenthesis
TS_LPAREN           = '('                # Left Parenthesis Symbol

TT_RPAREN           = 'RPAREN'           # Right Parenthesis
TS_RPAREN           = ')'                # Right Parenthesis Symbol

TT_EQUALS           = 'EQUALS'           # Equals
TS_EQUALS           = '='                # Equals Symbol


# Logical Operators
TT_NOT              = 'NOT'              # Not
TS_NOT              = '!'                # Not symbol

TT_AND              = 'AND'              # And
TS_AND              = '&'                # And symbol

TT_OR               = 'OR'               # Or
TS_OR               = '|'                # Or symbol


# Comparison Operators
TT_COMP_EE          = 'COMP_EE'          # Compare Equal
TS_COMP_EE          = '=='               # Compare Equal symbol

TT_COMP_NE          = 'COMP_NE'          # Compare Not Equal
TS_COMP_NE          = '!='               # Compare Not Equal symbol

TT_COMP_LT          = 'COMP_LT'          # Compare Less Than
TS_COMP_LT          = '<'                # Compare Less Than symbol

TT_COMP_GT          = 'COMP_GT'          # Compare Greater Than
TS_COMP_GT          = '>'                # Compare Greater Than symbol

TT_COMP_LTE         = 'COMP_LTE'         # Compare Less Than or Equal To
TS_COMP_LTE         = '<='               # Compare Less Than or Equal To symbol

TT_COMP_GTE         = 'COMP_GTE'         # Compare Greater Than or Equal To
TS_COMP_GTE         = '>='               # Compare Greater Than or Equal To symbol


# Other
TT_IDENT            = 'IDENT'            # Indentifier (i.e. variable name)

TT_KEYWORD          = 'KEYWORD'          # Keyword (reserved identifier)

TT_EOF              = 'EOF'              # End of File


# Keywords
KEY_VAR             = 'var'              # Variable keyword
KEY_AND             = 'and'              # And keyword
KEY_OR              = 'or'               # Or keyword
KEY_NOT             = 'not'              # Not keyword

KEYWORDS = [
    KEY_VAR,
    KEY_AND,
    KEY_OR,
    KEY_NOT
]


class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end=None):
        self.type = type_
        self.value = value

        if pos_start:
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end:
            self.pos_end = pos_end

    # 'Pretty' print format
    def __repr__(self):
        if self.value: return f'{self.type}:{self.value}'
        return f'{self.type}'

    def matches (self, type_, value):
        return self.type == type_ and self.value == value

################################################################
# LEXER
################################################################

class Lexer:
    def __init__(self, filename, text):
        self.filename = filename
        self.text = text
        self.pos = Position(-1, 0, -1, filename, text)
        self.current_char = None
        self.advance()
    
    # Advance to the next character in the text and set current_char to it, or None if at the end of the text
    def advance (self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.index] if self.pos.index < len(self.text) else None

    # Convert the text into an array of tokens
    def make_tokens(self):
        tokens = []

        while self.current_char != None:

            # Ignore spaces and tabs
            if self.current_char in ' \t':
                self.advance()

            # Digits
            if self.current_char in DIGITS:
                tokens.append(self.make_number())

            # Letters/identifiers/keywords
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())

            # Plus
            elif self.current_char == TS_PLUS:
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()

            # Minus
            elif self.current_char == TS_MINUS:
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()

            # Multiply
            elif self.current_char == TS_MULT:
                tokens.append(Token(TT_MULT, pos_start=self.pos))
                self.advance()

            # Divide
            elif self.current_char == TS_DIV:
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()

            # Power
            elif self.current_char == TS_POW:
                tokens.append(Token(TT_POW, pos_start=self.pos))
                self.advance()

            # Left Parenthesis
            elif self.current_char == TS_LPAREN:
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()

            # Right Parenthesis
            elif self.current_char == TS_RPAREN:
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()

            # Not Equals
            elif self.current_char == TS_NOT:
                tokens.append(self.make_not_equals())

            # Equals
            elif self.current_char == TS_EQUALS:
                tokens.append(self.make_equals())
                
            # Less Than
            elif self.current_char == TS_COMP_LT:
                tokens.append(self.make_less_than())

            # Greater Than
            elif self.current_char == TS_COMP_GT:
                tokens.append(self.make_greater_than())

            # And
            elif self.current_char == TS_AND:
                token, error = self.make_and()
                if error: return [], error
                tokens.append(token)

            # Or
            elif self.current_char == TS_OR:
                token, error = self.make_or()
                if error: return [], error
                tokens.append(token)

            # Error: Illegal Character
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None

    # Parse and convert input text into a valid number (integer or float)
    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + '.':
            # If a dot, increment the dot count and add it to the string so we know this is a float and not an integer
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

    # Parse input text into a valid identifier or keyword (i.e. name of variable)
    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in IDENTIFIER_CHARS:
            id_str += self.current_char
            self.advance()

        # If not a keyword, return as an identifier
        if id_str not in KEYWORDS:
            return Token(TT_IDENT, id_str, pos_start, self.pos)
        
        # If the keyword is the AND keyword, return as TT_AND
        if id_str == KEY_AND:
            return Token(TT_AND, pos_start=pos_start, pos_end=self.pos)
        
        # If the keyword is the OR keyword, return as TT_OR
        elif id_str == KEY_OR:
            return Token(TT_OR, pos_start=pos_start, pos_end=self.pos)

        # If the keyword is the NOT keyword, return as TT_NOT
        elif id_str == KEY_NOT:
            return Token(TT_NOT, pos_start=pos_start, pos_end=self.pos)
        
        # Otherwise return as a generic keyword
        return Token(TT_KEYWORD, id_str, pos_start, self.pos)

    def make_not_equals(self):
        token_type = TT_NOT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == TS_EQUALS:
            self.advance()
            token_type = TT_COMP_NE

        return Token(token_type, pos_start=pos_start, pos_end=self.pos)

    def make_equals(self):
        token_type = TT_EQUALS
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == TS_EQUALS:
            self.advance()
            token_type = TT_COMP_EE

        return Token(token_type, pos_start=pos_start, pos_end=self.pos)

    def make_less_than(self):
        token_type = TT_COMP_LT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == TS_EQUALS:
            self.advance()
            token_type = TT_COMP_LTE

        return Token(token_type, pos_start=pos_start, pos_end=self.pos)

    def make_greater_than(self):
        token_type = TT_COMP_GT
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == TS_EQUALS:
            self.advance()
            token_type = TT_COMP_GTE

        return Token(token_type, pos_start=pos_start, pos_end=self.pos)

    def make_and(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == TS_AND:
            self.advance()
            return Token(TT_AND, pos_start=pos_start, pos_end=self.pos), None

        self.advance()
        return None, ExpectedCharError(pos_start, self.pos, f'Expected "{TS_AND}" after "{TS_AND}".')

    def make_or(self):
        pos_start = self.pos.copy()
        self.advance()

        if self.current_char == TS_OR:
            self.advance()
            return Token(TT_OR, pos_start=pos_start, pos_end=self.pos), None

        self.advance()
        return None, ExpectedCharError(pos_start, self.pos, f'Expected "{TS_OR}" after "{TS_OR}".')

################################################################
# NODES
################################################################
class NumberNode:
    def __init__(self, token):
        self.token = token

        self.pos_start = self.token.pos_start
        self.pos_end = self.token.pos_end

    def __repr__(self):
        return f'{self.token}'

class VarAccessNode:
    def __init__(self, var_name_token):
        self.var_name_token = var_name_token

        self.pos_start = self.var_name_token.pos_start
        self.pos_end = self.var_name_token.pos_end

class VarAssignNode:
    def __init__(self, var_name_token, value_node):
        self.var_name_token = var_name_token
        self.value_node = value_node

        self.pos_start = self.var_name_token.pos_start
        self.pos_end = self.value_node.pos_end

class BinaryOpNode:
    def __init__(self, left_node, op, right_node):
        self.left_node = left_node
        self.op = op
        self.right_node = right_node

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op}, {self.right_node})'

class UnaryOpNode:
    def __init__(self, op, node):
        self.op = op
        self.node = node

        self.pos_start = self.op.pos_start
        self.pos_end = node.pos_end

    def __repr__(self):
        return f'({self.op}, {self.node})'

################################################################
# PARSER
################################################################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0

    def register_advancement(self):
        self.advance_count += 1
    
    def register(self, res):
        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node

    def success(self, node):
        self.node = node
        return self

    def failure(self, error):
        if not self.error or self.advance_count == 0:
            self.error = error
        return self

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.index = -1
        self.advance()

    def advance(self):
        self.index += 1
        if self.index < len(self.tokens):
            self.current_token = self.tokens[self.index]
        return self.current_token

    # Parse the given text into a parse tree with operations that our lanuage can execute. See grammar.txt for grammar details.
    def parse(self):
        res = self.expression()

        if not res.error and self.current_token.type != TT_EOF:
            return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, f'Expected "{TS_PLUS}", "{TS_MINUS}", "{TS_MULT}", "{TS_DIV}", or "{TS_POW}"'))
        
        return res

    # An atom is our highest priority node representing a simple number, expression wrapped in parenthesis, or an identifier (variable)
    def atom(self):
        res = ParseResult()
        token = self.current_token

        # If the current token is an integer or float number, return a number node
        if token.type in (TT_INT, TT_FLOAT):
            res.register_advancement()
            self.advance()
            return res.success(NumberNode(token))

        # If the current token is an identifier, return a variable access node
        elif token.type == TT_IDENT:
            res.register_advancement()
            self.advance()
            return res.success(VarAccessNode(token))

        # If the current token is a left parenthesis, then we'll expect an expression followed by a right parenthesis
        elif token.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            expression = res.register(self.expression())
            if res.error: return res

            if self.current_token.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
                return res.success(expression)
            else:
                return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, f'Expected "{TS_RPAREN}". Missing closing parenthesis.'))

        return res.failure(InvalidSyntaxError(token.pos_start, token.pos_end, f'Expected int, float, indentifier, "{TS_PLUS}", "{TS_MINUS}", or "{TS_LPAREN}"'))

    # A power is a node that represents an atom raised to an exponent term
    def power(self):
        return self.binary_operation(self.atom, (TT_POW, ), self.factor)

    # A factor represents a power or a modified unary factor (positive or negative before the factor)
    def factor(self):
        res = ParseResult()
        token = self.current_token

        # If the current token is a plus or minus we will expect a unary operation (+ or -) before another factor (typically a number). i.e. -5
        if token.type in (TT_PLUS, TT_MINUS):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(token, factor))

        return self.power()

    # A term represents factors that are multiplied or divided together
    def term(self):
        return self.binary_operation(self.factor, (TT_MULT, TT_DIV))

    # An arithmetic expression represents binary terms that are added or subtracted together
    def arithmetic_expression(self):
        return self.binary_operation(self.term, (TT_PLUS, TT_MINUS))

    # A comparison expression represents binary arithmetic expressions with comparison operators (equals, not equal, greater than, etc) or logical negations 
    def comparison_expression(self):
        res = ParseResult()

        # If the current token is a 'not' (negation) token, create a unary operation node to negate the comparison_expression after the negation token
        if self.current_token.type == TT_NOT:
            op_token = self.current_token
            res.register_advancement()
            self.advance()

            node = res.register(self.comparison_expression())
            if res.error: return res
            return res.success(UnaryOpNode(op_token, node))

        # Otherwise return a binary operation of twp arithmetic expressions with compariosn operators
        node = res.register(self.binary_operation(self.arithmetic_expression, (TT_COMP_EE, TT_COMP_NE, TT_COMP_LT, TT_COMP_GT, TT_COMP_LTE, TT_COMP_GTE)))
        if res.error:
            return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, f'Expected int, float, indentifier, "{TS_PLUS}", "{TS_MINUS}", "{TS_LPAREN}".'))

        return res.success(node)

    # An expression represents binary comparison expressions or a variable assignment
    def expression(self):
        res = ParseResult()

        # Check that the current token is a 'var' keyword
        if self.current_token.matches(TT_KEYWORD, KEY_VAR):
            res.register_advancement()
            self.advance()

            # Check that the token after the 'var' keyword is an identifier
            if self.current_token.type != TT_IDENT:
                return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, "Expected identifier."))

            var_name = self.current_token
            res.register_advancement()
            self.advance()

            # Check that the token after the identifier is an equals operator
            if self.current_token.type != TT_EQUALS:
                return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, f'Expected "{TS_EQUALS}" operator.'))

            res.register_advancement()
            self.advance()
            expression = res.register(self.expression())

            if res.error: return res
            return res.success(VarAssignNode(var_name, expression))

        # Return a binary operation of two comparison expressions with logical keywords (AND, OR)
        node = res.register(self.binary_operation(self.comparison_expression, (TT_AND, TT_OR)))

        if res.error:
            return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, f'Expected int, float, indentifier, "{KEY_VAR}", "{TS_PLUS}", "{TS_MINUS}", or "{TS_RPAREN}"'))

        return res.success(node)

    def binary_operation(self, func_a, ops, func_b=None):
        if func_b == None:
            func_b = func_a

        res = ParseResult()
        left = res.register(func_a())

        if res.error: return res

        while self.current_token.type in ops or (self.current_token.type, self.current_token.value) in ops:
            op = self.current_token
            res.register_advancement()
            self.advance()
            right = res.register(func_b())

            if res.error: return res
            
            left = BinaryOpNode(left, op, right)
        
        return res.success(left)

################################################################
# VALUES
################################################################

class Number:
    def __init__(self, value):
        self.value = value
        self.set_pos()
        self.set_context()
    
    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def add_to(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None

    def subtract_by(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None

    def multiply_by(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None

    def divide_by(self, other):
        if isinstance(other, Number):
            # Check for divide by 0 error
            if other.value == 0:
                return None, RuntimeError(other.pos_start, other.pos_end, "Division by zero.", self.context)

            return Number(self.value / other.value).set_context(self.context), None

    def powered_by(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None

    def anded_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value and other.value)).set_context(self.context), None

    def ored_by(self, other):
        if isinstance(other, Number):
            return Number(int(self.value or other.value)).set_context(self.context), None

    def notted(self):
        return Number(1 if self.value == 0 else 0).set_context(self.context), None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return str(self.value)

################################################################
# INTERPRETER
################################################################

class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None
    
    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set(self, name, value):
        self.symbols[name] = value

    def remove(self, name):
        del self.symbols[name]

class RuntimeResult:
    def __init__(self):
        self.value = None
        self.error = None
    
    def register(self, res):
        if res.error: self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self

class Interpreter:
    def visit(self, node, context):
        # Create a string to represent our node using it's node type/name. i.e. a binary operation would result in "visit_BinaryOpNode"
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)

    def no_visit_method(self, node, context):
        raise Exception(f'No visit_{type(node).__name__} method defined')

    def visit_NumberNode(self, node, context):
        return RuntimeResult().success(Number(node.token.value).set_context(context).set_pos(node.pos_start, node.pos_end))

    def visit_VarAccessNode(self, node, context):
        res = RuntimeResult()
        var_name = node.var_name_token.value
        value = context.symbol_table.get(var_name)

        if not value:
            return res.failure(RuntimeError(node.pos_start, node.pos_end, f'"{var_name}" is not defined.', context))
        
        value = value.copy().set_pos(node.pos_start, node.pos_end)
        return res.success(value)

    def visit_VarAssignNode(self, node, context):
        res = RuntimeResult()
        var_name = node.var_name_token.value
        value = res.register(self.visit(node.value_node, context))
        if res.error: return res

        context.symbol_table.set(var_name, value)
        
        return res.success(value)

    def visit_BinaryOpNode(self, node, context):
        res = RuntimeResult()
        left = res.register(self.visit(node.left_node, context))
        if res.error: return res
        right = res.register(self.visit(node.right_node, context))
        if res.error: return res

        if node.op.type == TT_PLUS:
            result, error = left.add_to(right)
        elif node.op.type == TT_MINUS:
            result, error = left.subtract_by(right)
        elif node.op.type == TT_MULT:
            result, error = left.multiply_by(right)
        elif node.op.type == TT_DIV:
            result, error = left.divide_by(right)
        elif node.op.type == TT_POW:
            result, error = left.powered_by(right)
        elif node.op.type == TT_COMP_EE:
            result, error = left.get_comparison_eq(right)
        elif node.op.type == TT_COMP_NE:
            result, error = left.get_comparison_ne(right)
        elif node.op.type == TT_COMP_LT:
            result, error = left.get_comparison_lt(right)
        elif node.op.type == TT_COMP_GT:
            result, error = left.get_comparison_gt(right)
        elif node.op.type == TT_COMP_LTE:
            result, error = left.get_comparison_lte(right)
        elif node.op.type == TT_COMP_GTE:
            result, error = left.get_comparison_gte(right)
        elif node.op.type == TT_AND:
            result, error = left.anded_by(right)
        elif node.op.type == TT_OR:
            result, error = left.ored_by(right)

        if error:
            return res.failure(error)
        else:
            return res.success(result.set_pos(node.pos_start, node.pos_end))

    def visit_UnaryOpNode(self, node, context):
        res = RuntimeResult()
        number = res.register(self.visit(node.node, context))
        if res.error: return res

        error = None

        if node.op.type == TT_MINUS:
            number, error = number.multiply_by(Number(-1))
        elif node.op.type == TT_NOT:
            number, error = number.notted()

        if error:
            return res.failure(error)
        else:
            return res.success(number.set_pos(node.pos_start, node.pos_end))

################################################################
# RUN
################################################################

global_symbol_table = SymbolTable()
global_symbol_table.set("null", Number(0))
global_symbol_table.set("true", Number(1))
global_symbol_table.set("false", Number(0))

def run(filename, text):
    lexer = Lexer(filename, text)
    tokens, error = lexer.make_tokens()

    if error: return None, error

    # Generate the abstract syntax tree (ast)
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    # Run program
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error

def test_lexer(filename, text):
    lexer = Lexer(filename, text)
    tokens, error = lexer.make_tokens()
    return tokens, error

def test_parser(filename, text):
    lexer = Lexer(filename, text)
    tokens, error = lexer.make_tokens()

    # Generate the abstract syntax tree (ast)
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    return ast.node, None