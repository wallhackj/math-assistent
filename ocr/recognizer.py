def transform_to_latex(handwritten_text):
    latex_symbols_dict = {
        '-' : '-',',' : ',','!' : '!','(' : '(',')' : ')',
        '[' : '[',']' : ']','{' : '{','}' : '}','+' : '+','=' : '=',
        '0' : '0','1' : '1','2' : '2','3' : '3','4' : '4','5' : '5',
        '6' : '6','7' : '7','8' : '8','9' : '9','A' : 'A','alpha' : '\\alpha',
        'ascii_124' : '|','b' : 'b','beta' : '\\beta','C' : 'C','cos' : '\\cos','d' : 'd',
        'Delta' : '\\delta','div' : '\\div','e' : 'e','exists' : '\\exists','f' : 'f','\\forall' : 'forall',
        'forward_slash' : '/','G' : 'G','gamma' : '\\gamma','geq' : '\\geq','gt' : '>','H' : 'H',
        'i' : 'i','in' : '\\in','infty' : '\\infty','int' : '\\int','j' : 'j','k' : 'k',
        'l' : 'l','lambda' : '\\lambda','ldots' : '\\ldots','leq' : '\leq','\\lim' : 'lim','log' : '\log',
        'lt':'<','M':'M','mu':'\mu','N':'N','neq':'\\neq','o':'o','p':'p','phi':'\\phi','pi':'\\pi',
        'pm':'\\pm','prime':'\\prime','q':'q','R':'R','rightarrow':'\Rightarrow','S':'S','sigma':'\\sigma',
        'sin':'\\sin','sqrt':'\\sqrt','sum':'\sum','T':'T','tan':'\\tan','theta':'\\theta','times':'\\times',
        'u':'u','v':'v','w':'w','X':'X','y':'y','z':'z'
    }
    return latex_symbols_dict.get(handwritten_text)