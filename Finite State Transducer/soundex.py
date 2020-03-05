# Name: Lizzie Liang
# HW1
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
                        'B':'1', 'F':'1', 'P':'1' , 'V':'1', 
                        'c':'2', 'g':'2', 'j':'2', 'k':'2', 'q':'2', 's':'2', 'x':'2', 'z':'2',
                        'C':'2', 'G':'2', 'J':'2', 'K':'2', 'Q':'2', 'S':'2', 'X':'2', 'Z':'2',
                        'd':'3', 't':'3', 'D':'3', 'T': '3',
                        'l':'4', 'L':'4',
                        'm':'5', 'n': '5', 'M':'5', 'N': '5',
                        'r': '6', 'R':'6'
                         }
    
    # Let's define our first FST
    f1 = FST('soundex-generate')

    # Indicate that 'start' is the initial state
    #two nodes for each group. First node is to retain the first letter 
    #The second node is to 
    f1.add_state('s')
    f1.add_state('11') 
    f1.add_state('21')
    f1.add_state('31')
    f1.add_state('41')
    f1.add_state('51')
    f1.add_state('61')
    f1.add_state('71') # you only need one node for vowel group
    f1.add_state('12')
    f1.add_state('22')
    f1.add_state('32')
    f1.add_state('42')
    f1.add_state('52')
    f1.add_state('62')
    
    f1.initial_state = 's'

    # Set all the final states
    f1.set_final('11')
    f1.set_final('12')
    f1.set_final('21')
    f1.set_final('22')
    f1.set_final('31')
    f1.set_final('32')
    f1.set_final('41')
    f1.set_final('42')
    f1.set_final('51')
    f1.set_final('52')
    f1.set_final('61')
    f1.set_final('62')
    f1.set_final('71')
            
    #remove all non-initial occurrences of vowels - a, e, h, i, o, u, w, y
    for letter in ['a','e','h','i','o','u','w','y', 'A','E','H','I','O','U','W','Y']:
        f1.add_arc('71', '71', letter, '')
        f1.add_arc('12', '71', letter, '')
        f1.add_arc('22', '71', letter, '')
        f1.add_arc('32', '71', letter, '')
        f1.add_arc('42', '71', letter, '')
        f1.add_arc('52', '71', letter, '')
        f1.add_arc('62', '71', letter, '')
        f1.add_arc('11', '71', letter, '')
        f1.add_arc('21', '71', letter, '')
        f1.add_arc('31', '71', letter, '')
        f1.add_arc('41', '71', letter, '')
        f1.add_arc('51', '71', letter, '')
        f1.add_arc('61', '71', letter, '')
        f1.add_arc('s', '71', letter, letter)
        

    #replace the letters to numbers according to the mapping
    #if two or more letters from the same number group were adjacent 
    #in the original name, only replace the first one
    #group 1
    for key in {'b':'1', 'f':'1', 'p':'1' , 'v':'1', 'B':'1', 'F':'1', 'P':'1' , 'V':'1'}:
        f1.add_arc('s', '11', key, key) #to retrain the first letter
        f1.add_arc('21', '11', key, letter_to_number[key])
        f1.add_arc('22', '11', key, letter_to_number[key])
        f1.add_arc('31', '11', key, letter_to_number[key])
        f1.add_arc('32', '11', key, letter_to_number[key])
        f1.add_arc('41', '11', key, letter_to_number[key])
        f1.add_arc('42', '11', key, letter_to_number[key])
        f1.add_arc('51', '11', key, letter_to_number[key])
        f1.add_arc('52', '11', key, letter_to_number[key])
        f1.add_arc('61', '11', key, letter_to_number[key])
        f1.add_arc('62', '11', key, letter_to_number[key])
        f1.add_arc('71', '11', key, letter_to_number[key])
        f1.add_arc('11', '12', key, '')
        f1.add_arc('12', '12', key, '')
        
    
    #group 2    
    for key in {'c':'2', 'g':'2', 'j':'2', 'k':'2', 'q':'2', 's':'2', 'x':'2', 'z':'2',
                'C':'2', 'G':'2', 'J':'2', 'K':'2', 'Q':'2', 'S':'2', 'X':'2', 'Z':'2'}:
        f1.add_arc('s', '21', key, key) #to retrain the first letter
        f1.add_arc('11', '21', key, letter_to_number[key])
        f1.add_arc('12', '21', key, letter_to_number[key])
        f1.add_arc('31', '21', key, letter_to_number[key])
        f1.add_arc('32', '21', key, letter_to_number[key])
        f1.add_arc('41', '21', key, letter_to_number[key])
        f1.add_arc('42', '21', key, letter_to_number[key])
        f1.add_arc('51', '21', key, letter_to_number[key])
        f1.add_arc('52', '21', key, letter_to_number[key])
        f1.add_arc('61', '21', key, letter_to_number[key])
        f1.add_arc('62', '21', key, letter_to_number[key])
        f1.add_arc('71', '21', key, letter_to_number[key])
        f1.add_arc('21', '22', key, '')
        f1.add_arc('22', '22', key, '')
        
    
    #group 3    
    for key in {'d':'3', 't':'3', 'D':'3', 'T':'3'}:
        f1.add_arc('s', '31', key, key) #to retrain the first letter
        f1.add_arc('21', '31', key, letter_to_number[key])
        f1.add_arc('22', '31', key, letter_to_number[key])
        f1.add_arc('11', '31', key, letter_to_number[key])
        f1.add_arc('12', '31', key, letter_to_number[key])
        f1.add_arc('41', '31', key, letter_to_number[key])
        f1.add_arc('42', '31', key, letter_to_number[key])
        f1.add_arc('51', '31', key, letter_to_number[key])
        f1.add_arc('52', '31', key, letter_to_number[key])
        f1.add_arc('61', '31', key, letter_to_number[key])
        f1.add_arc('62', '31', key, letter_to_number[key])
        f1.add_arc('71', '31', key, letter_to_number[key])
        f1.add_arc('31', '32', key, '')
        f1.add_arc('32', '32', key, '')

    #group 4        
    for key in {'l':'4', 'L':'4'}:
        f1.add_arc('s', '41', key, key) #to retrain the first letter
        f1.add_arc('21', '41', key, letter_to_number[key])
        f1.add_arc('22', '41', key, letter_to_number[key])
        f1.add_arc('31', '41', key, letter_to_number[key])
        f1.add_arc('32', '41', key, letter_to_number[key])
        f1.add_arc('11', '41', key, letter_to_number[key])
        f1.add_arc('12', '41', key, letter_to_number[key])
        f1.add_arc('51', '41', key, letter_to_number[key])
        f1.add_arc('52', '41', key, letter_to_number[key])
        f1.add_arc('61', '41', key, letter_to_number[key])
        f1.add_arc('62', '41', key, letter_to_number[key])
        f1.add_arc('71', '41', key, letter_to_number[key])
        f1.add_arc('41', '42', key, '')
        f1.add_arc('42', '42', key, '')

    #group 5        
    for key in {'m':'5', 'n': '5', 'M':'5', 'N': '5'}:
        f1.add_arc('s', '51', key, key) #to retrain the first letter
        f1.add_arc('21', '51', key, letter_to_number[key])
        f1.add_arc('22', '51', key, letter_to_number[key])
        f1.add_arc('31', '51', key, letter_to_number[key])
        f1.add_arc('32', '51', key, letter_to_number[key])
        f1.add_arc('41', '51', key, letter_to_number[key])
        f1.add_arc('42', '51', key, letter_to_number[key])
        f1.add_arc('11', '51', key, letter_to_number[key])
        f1.add_arc('12', '51', key, letter_to_number[key])
        f1.add_arc('61', '51', key, letter_to_number[key])
        f1.add_arc('62', '51', key, letter_to_number[key])
        f1.add_arc('71', '51', key, letter_to_number[key])
        f1.add_arc('51', '52', key, '')
        f1.add_arc('52', '52', key, '')

    #group 6        
    for key in {'r': '6', 'R': '6'}:
        f1.add_arc('s', '61', key, key) #to retrain the first letter
        f1.add_arc('21', '61', key, letter_to_number[key])
        f1.add_arc('22', '61', key, letter_to_number[key])
        f1.add_arc('31', '61', key, letter_to_number[key])
        f1.add_arc('32', '61', key, letter_to_number[key])
        f1.add_arc('41', '61', key, letter_to_number[key])
        f1.add_arc('42', '61', key, letter_to_number[key])
        f1.add_arc('51', '61', key, letter_to_number[key])
        f1.add_arc('52', '61', key, letter_to_number[key])
        f1.add_arc('11', '61', key, letter_to_number[key])
        f1.add_arc('12', '61', key, letter_to_number[key])
        f1.add_arc('71', '61', key, letter_to_number[key])
        f1.add_arc('61', '62', key, '')
        f1.add_arc('62', '62', key, '')

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
