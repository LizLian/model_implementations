import unittest
from soundex import letters_to_numbers, truncate_to_three_digits, add_zero_padding, soundex_convert
from morphology import Parser

class TestHW1(unittest.TestCase):

    def setUp(self):
        self.f1 = letters_to_numbers()
        self.f2 = truncate_to_three_digits()
        self.f3 = add_zero_padding()
        self.mparser = Parser()

    def test_letters(self):
        self.assertEqual("".join(self.f1.transduce("washington")[0]), "w25235")
        self.assertEqual("".join(self.f1.transduce("jefferson")[0]), "j1625")
        self.assertEqual("".join(self.f1.transduce("adams")[0]), "a352")
        self.assertEqual("".join(self.f1.transduce("bush")[0]), "b2")

    def test_truncation(self):
        self.assertEqual("".join(self.f2.transduce("a33333")[0]), "a333")
        self.assertEqual("".join(self.f2.transduce("123456")[0]), "123")
        self.assertEqual("".join(self.f2.transduce("11")[0]), "11")
        self.assertEqual("".join(self.f2.transduce("5")[0]), "5")
        self.assertEqual("".join(self.f2.transduce("a")[0]), "a")
        self.assertEqual("".join(self.f2.transduce("0")[0]), "0")
        self.assertEqual("".join(self.f2.transduce("a12a")[0]), "a12a")
        self.assertEqual("".join(self.f2.transduce("a12a1234")[0]), "a12a123")

    def test_padding(self):
        self.assertEqual("".join(self.f3.transduce("3")[0]), "300")
        self.assertEqual("".join(self.f3.transduce("30")[0]), "300")
        self.assertEqual("".join(self.f3.transduce("b56")[0]), "b560")
        self.assertEqual("".join(self.f3.transduce("c111")[0]), "c111")
        self.assertEqual("".join(self.f3.transduce("a")[0]), "a000")
        self.assertEqual("".join(self.f3.transduce("0")[0]), "000")

    def test_soundex(self):
        self.assertEqual(soundex_convert("jurafsky"), "j612")
        self.assertEqual(soundex_convert("Lukaschowsky"), "L222")
        self.assertEqual(soundex_convert("Roses"), "R220")
        self.assertEqual(soundex_convert("Tomzak"), "T522")
        self.assertEqual(soundex_convert("Pfister"), "P236")
        self.assertEqual(soundex_convert("Lloyd"), "L300")
        self.assertEqual(soundex_convert("Ashcroft"), "A226")

    def test_morphology(self):
        havocking = list('havocking')
        self.assertEqual(self.mparser.parse(havocking), "havoc+present participle form")
        havocking = list('havocked')
        self.assertEqual(self.mparser.parse(havocking), "havoc+past form")
        havocking = list('licking')
        self.assertEqual(self.mparser.parse(havocking), "lick+present participle form")
        havocking = list('licked')
        self.assertEqual(self.mparser.parse(havocking), "lick+past form")
        havocking = list('wanting')
        self.assertEqual(self.mparser.parse(havocking), "want+present participle form")
        havocking = list('wanted')
        self.assertEqual(self.mparser.parse(havocking), "want+past form")
        havocking = list('panicking')
        self.assertEqual(self.mparser.parse(havocking), "panic+present participle form")
        havocking = list('panicked')
        self.assertEqual(self.mparser.parse(havocking), "panic+past form")
        havocking = list('syncing')
        self.assertEqual(self.mparser.parse(havocking), "sync+present participle form")
        havocking = list('synced')
        self.assertEqual(self.mparser.parse(havocking), "sync+past form")

        lick = ['l','i','c','k','+past form']
        self.assertEqual(self.mparser.generate(lick), "licked")
        panic_past = ['p', 'a', 'n', 'i', 'c', '+past form']
        self.assertEqual(self.mparser.generate(panic_past), "panicked")
        panic_present = ['p', 'a', 'n', 'i', 'c', '+present participle form']
        self.assertEqual(self.mparser.generate(panic_present), "panicking")
        want_present = ['w', 'a', 'n', 't', '+present participle form']
        self.assertEqual(self.mparser.generate(want_present), "wanting")
        want_past = ['w', 'a', 'n', 't', '+past form']
        self.assertEqual(self.mparser.generate(want_past), "wanted")
        sync_present = ['s', 'y', 'n', 'c', '+present participle form']
        self.assertEqual(self.mparser.generate(sync_present), "syncing")
        sync_past = ['s', 'y', 'n', 'c', '+past form']
        self.assertEqual(self.mparser.generate(sync_past), "synced")
        havoc_present = ['h', 'a', 'v', 'o', 'c', '+present participle form']
        self.assertEqual(self.mparser.generate(havoc_present), "havocking")
        havoc_past = ['h', 'a', 'v', 'o', 'c', '+past form']
        self.assertEqual(self.mparser.generate(havoc_past), "havocked")
if __name__ == '__main__':
    unittest.main()
