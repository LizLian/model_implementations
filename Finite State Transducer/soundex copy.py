from fst import FST
import string, sys
from fsmutils import compose

def letters_to_numbers():
    """
    Returns an FST that converts letters to numbers as specified by
    the soundex algorithm
    """
    #letter to number mapping
    letter_to_number = {'b':'1', 'f':'1', 'p':'1' , 'v':'1', 
                        'c':'2', 'g':'2', 'j':'2', 'k':'2', 'q':'2', 's':'2', 'x':'2', 'z':'2',
                        'd':'3', 't':'3',
                        'l':'4',
                        'm':'5', 'n': '5',
                        'r': '6'
                         }
    
    # Let's define our first FST
    f1 = FST('soundex-generate')

    # Indicate that 'start' is the initial state
    f1.add_state('start')
    f1.add_state('1')
    f1.add_state('2')
    f1.add_state('3')
    f1.add_state('4')
    f1.add_state('5')
    f1.add_state('6')
    f1.add_state('7')

    f1.initial_state = 'start'

    # Set all the final states
    f1.set_final('2')
    f1.set_final('1')
    f1.set_final('3')
    f1.set_final('4')
    f1.set_final('5')
    f1.set_final('6')
    f1.set_final('7')
    
    #retain the first letter
    for letter in list(string.ascii_letters):
        f1.add_arc('start', '1', letter, letter)
        f1.add_arc('2', '1', letter, '')
        f1.add_arc('3', '1', letter, '')
        f1.add_arc('4', '1', letter, '')
        f1.add_arc('5', '1', letter, '')
        f1.add_arc('6', '1', letter, '')
        f1.add_arc('7', '1', letter, '')
        f1.add_arc('1', '1', letter, '')
            
    #remove all non-initial occurrences of a, e, h, i, o, u, w, y
    for letter in ['a','e','h','i','o','u','w','y', 'A','E','H','I','O','U','W','Y']:
        
        f1.add_arc('2', '1', letter, '')
        f1.add_arc('3', '1', letter, '')
        f1.add_arc('4', '1', letter, '')
        f1.add_arc('5', '1', letter, '')
        f1.add_arc('6', '1', letter, '')
        f1.add_arc('7', '1', letter, '')

    #replace the letters to numbers according to the mapping
    #if two or more letters from the same number group were adjacent 
    #in the original name, only replace the first one
    for key in {'b':'1', 'f':'1', 'p':'1' , 'v':'1'}:
        f1.add_arc('1', '2', key, letter_to_number[key])
        f1.add_arc('2', '1', key, '')
        f1.add_arc('3', '2', key, letter_to_number[key])
        f1.add_arc('4', '2', key, letter_to_number[key])
        f1.add_arc('5', '2', key, letter_to_number[key])
        f1.add_arc('6', '2', key, letter_to_number[key])
        f1.add_arc('7', '2', key, letter_to_number[key])
        
    for key in {'c':'2', 'g':'2', 'j':'2', 'k':'2', 'q':'2', 's':'2', 'x':'2', 'z':'2'}:
        f1.add_arc('1', '3', key, letter_to_number[key])
        f1.add_arc('3', '1', key, '')
        f1.add_arc('2', '3', key, letter_to_number[key])
        f1.add_arc('4', '3', key, letter_to_number[key])
        f1.add_arc('5', '3', key, letter_to_number[key])
        f1.add_arc('6', '3', key, letter_to_number[key])
        f1.add_arc('7', '3', key, letter_to_number[key])
        
    for key in {'d':'3', 't':'3'}:
        f1.add_arc('1', '4', key, letter_to_number[key])
        f1.add_arc('4', '1', key, '')
        f1.add_arc('3', '4', key, letter_to_number[key])
        f1.add_arc('2', '4', key, letter_to_number[key])
        f1.add_arc('5', '4', key, letter_to_number[key])
        f1.add_arc('6', '4', key, letter_to_number[key])
        f1.add_arc('7', '4', key, letter_to_number[key])
        
    for key in {'l':'4'}:
        f1.add_arc('1', '5', key, letter_to_number[key])
        f1.add_arc('5', '1', key, '')
        f1.add_arc('3', '5', key, letter_to_number[key])
        f1.add_arc('4', '5', key, letter_to_number[key])
        f1.add_arc('2', '5', key, letter_to_number[key])
        f1.add_arc('6', '5', key, letter_to_number[key])
        f1.add_arc('7', '5', key, letter_to_number[key])
        
    for key in {'m':'5', 'n': '5'}:
        f1.add_arc('1', '6', key, letter_to_number[key])
        f1.add_arc('6', '1', key, '')
        f1.add_arc('3', '6', key, letter_to_number[key])
        f1.add_arc('4', '6', key, letter_to_number[key])
        f1.add_arc('5', '6', key, letter_to_number[key])
        f1.add_arc('2', '6', key, letter_to_number[key])
        f1.add_arc('7', '6', key, letter_to_number[key])
        
    for key in {'r': '6'}:
        f1.add_arc('1', '7', key, letter_to_number[key])
        f1.add_arc('7', '1', key, '')
        f1.add_arc('3', '7', key, letter_to_number[key])
        f1.add_arc('4', '7', key, letter_to_number[key])
        f1.add_arc('5', '7', key, letter_to_number[key])
        f1.add_arc('6', '7', key, letter_to_number[key])
        f1.add_arc('2', '7', key, letter_to_number[key])

    return f1

def truncate_to_three_digits():
    """
    Create an FST that will truncate a soundex string to three digits
    """

    # Ok so now let's do the second FST, the one that will truncate
    # the number of digits to 3
    f2 = FST('soundex-truncate')

    # Indicate initial and final states
    f2.add_state('1')
    f2.add_state('2')
    f2.add_state('3')
    f2.add_state('4')
    f2.initial_state = '1'
    f2.set_final('1')
    f2.set_final('2')
    f2.set_final('3')
    f2.set_final('4')

    for letter in list(string.ascii_letters):
        f2.add_arc('1', '1', letter, letter)
        f2.add_arc('2', '1', letter, letter)
        f2.add_arc('3', '1', letter, letter)
        f2.add_arc('4', '1', letter, letter)
        
    for digit in ['1','2','3','4','5','6','7','8','9','0']:
        f2.add_arc('1', '2', digit, digit)
        f2.add_arc('2', '3', digit, digit)
        f2.add_arc('3', '4', digit, digit)
        f2.add_arc('4', '4', digit, '')
        
    return f2

def add_zero_padding():
    # Now, the third fst - the zero-padding fst
    f3 = FST('soundex-padzero')

    f3.add_state('1')
    f3.add_state('2')
    f3.add_state('3')
    f3.add_state('4')
    f3.add_state('5')
    f3.add_state('6')
    
    f3.initial_state = '1'
    f3.set_final('6')

    for letter in list(string.ascii_letters):
        f3.add_arc('1', '1', letter, letter)
    
    for digit in ['1','2','3','4','5','6','7','8','9','0']:
        f3.add_arc('1', '2', digit, digit)
        f3.add_arc('2', '5', digit, digit)
        f3.add_arc('5', '6', digit, digit)

    f3.add_arc('1','3', '', '0')
    f3.add_arc('3','4', '', '0')
    f3.add_arc('4','6', '', '0')
    f3.add_arc('2', '4', '', '0')
    f3.add_arc('5','6','','0')
    return f3

def soundex_convert(name_string):
    """Combine the three FSTs above and use it to convert a name into a Soundex"""
    f1 = letters_to_numbers()
    f1.transduce(list(name_string))
    f2 = truncate_to_three_digits()
    f3 = add_zero_padding()
    output = compose(name_string, f1, f2, f3)[0]
    return ''.join(output)

if __name__ == '__main__':
    # for Python 2, change input() to raw_input()
    user_input = input().strip()

    if user_input:
        print("%s -> %s" % (user_input, soundex_convert(list(user_input))))
