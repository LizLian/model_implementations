# Name: Lizzie Liang
# HW1
from fst import FST
import string
from fsmutils import compose

class Parser():

    def __init__(self):
        pass

    def generate(self, analysis):
        """Generate the morphologically correct word 

        e.g.
        p = Parser()
        analysis = ['p','a','n','i','c','+past form']
        p.generate(analysis) 
        ---> 'panicked'
        """

        #analysis = ['p','a','n','i','c','+past form']
        # Let's define our first FST

        f1 = FST('morphology-generate')
        
        f1.add_state('1')
        f1.add_state('2')
        f1.add_state('3')
        f1.add_state('4')
        f1.add_state('5') 
        f1.add_state('6') #non-c state
        f1.add_state('7') #c state
        f1.add_state('8') #add k
        f1.add_state('9') #+present 
        f1.add_state('10') #+past
        
        f1.initial_state = '1'
        #f1.set_final('8')
        f1.set_final('9')
        f1.set_final('10')
        
        #state 1 to 2, and 2 to 3. we don't care about vowel or consonant here
        for letter in list(string.ascii_letters):
            f1.add_arc('1', '2', letter, letter)
            f1.add_arc('2', '3', letter, letter)
        
        #3 to 5 input/output consonants
        vowels = ['a','e','i','o','u','A','E','I','O','U']
        consonants = [c for c in list(string.ascii_letters) if c not in vowels]
        non_c_con = [c for c in consonants if c not in ['c', 'C']]
        for letter in consonants:
            f1.add_arc('3', '5', letter, letter)
            f1.add_arc('5', '5', letter, letter)
        
        #the third and fourth input should be a vowel
        for letter in vowels:
            f1.add_arc('3', '4', letter, letter)
            f1.add_arc('4', '4', letter, letter)
        
        #if the fourth input is a non c consonant, go to 5
        for letter in non_c_con:
            f1.add_arc('4', '5', letter, letter)
            
        #if the input at state 5 is a vowel, go back to 4    
        for letter in vowels:
            f1.add_arc('5', '4', letter, letter)
        
        #if the second last letter is a c, go to 7
        f1.add_arc('4', '7', 'c', 'c')
        
        #add k after 7
        f1.add_arc('7', '8', '', 'k')
        #output nothing from 5 to 8
        f1.add_arc('5', '8', '', '')
        
        f1.add_arc('8','9','+present participle form','ing')
        f1.add_arc('8','10','+past form','ed')
        
        output = f1.transduce(analysis)[0]
        return ''.join(output)

    def parse(self, word):
        """Parse a word morphologically 

        e.g.
        p = Parser()
        word = ['p','a','n','i','c','k','i','n','g']
        p.parse(word)
        ---> 'panic+present participle form'
        """
        # Ok so now let's do the second FST
        f2 = FST('morphology-parse')
        f2.add_state('start')
        f2.initial_state = 'start'
               
        #add states for the word lick
        for w in list('lick'):
            state_name = 'lick-' + w
            f2.add_state(state_name)
        #add first letter    
        f2.add_arc('start', 'lick-l', 'l', 'l')
        
        #add arc for the word lick
        lick = list('lick')
        for w in range(0,len(lick)-1):
            f2.add_arc('lick-'+lick[w], 'lick-'+lick[w+1], lick[w+1], lick[w+1] )
        
        #add states for the word lick    
        for w in list('want'):
            state_name = 'want-' + w
            f2.add_state(state_name)
        
        f2.add_arc('start', 'want-w', 'w', 'w')
        #add arc for the word want
        want = list('want')
        for w in range(0,len(want)-1):
            f2.add_arc('want-'+want[w], 'want-'+want[w+1], want[w+1], want[w+1] )

        #add states for the word sync
        sync = list('sync')
        for w in sync:
            state_name = 'sync-' + w
            f2.add_state(state_name)
        
        f2.add_arc('start', 'sync-s', 's', 's')
        #add arc for the word sync
        for w in range(0,len(sync)-1):
            f2.add_arc('sync-'+sync[w], 'sync-'+sync[w+1], sync[w+1], sync[w+1] )
        
        #add states for the word panic
        panic = list('panic')
        for w in panic:
            state_name = 'panic-' + w
            f2.add_state(state_name)
        
        f2.add_arc('start', 'panic-p', 'p', 'p')
        #add arc for the word panic
        for w in range(0,len(panic)-1):
            f2.add_arc('panic-'+panic[w], 'panic-'+panic[w+1], panic[w+1], panic[w+1] )
        
        #add states for the word havoc
        havoc = list('havoc')
        for w in havoc:
            state_name = 'havoc-' + w
            f2.add_state(state_name)
            
        f2.add_arc('start', 'havoc-h', 'h', 'h')
        #add arc for the word havoc
        for w in range(0,len(havoc)-1):
            f2.add_arc('havoc-'+havoc[w], 'havoc-'+havoc[w+1], havoc[w+1], havoc[w+1] )
        
        f2.add_state('intermediate1')
        f2.add_state('intermediate2')
        f2.add_state('pres1')
        f2.add_state('past1')
        
        f2.add_arc('lick-k', 'intermediate1', '', '')
        f2.add_arc('want-t', 'intermediate1', '', '')
        f2.add_arc('sync-c', 'intermediate1', '', '')
        f2.add_arc('panic-c', 'intermediate1', 'k', '')
        f2.add_arc('havoc-c', 'intermediate1', 'k', '')
        
        f2.add_arc('intermediate1', 'pres1', 'ing', '+present participle form')
        f2.add_arc('intermediate1', 'past1', 'ed', '+past form')

        f2.set_final('pres1')
        f2.set_final('past1')
        
        if ''.join(word[-3:]) == 'ing':
            inputs = word[:-3]
            inputs.append('ing')
        elif ''.join(word[-2:]) == 'ed':
            inputs = word[:-2]
            inputs.append('ed')
        else:
            inputs = word
        
        output = f2.transduce(inputs)[0]
        return ''.join(output)
