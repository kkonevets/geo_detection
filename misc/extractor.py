from itertools import dropwhile
import unittest


def extract_city(row):
    for vs in row[1:]:  # extid
        candidates = []
        for v in vs.split(';'):
            v = v.strip()
            if not v:
                continue

            splited = v.split('.')
            if len(splited) < 3:
                continue

            temp = dropwhile(lambda s: not s.startswith(('ci_', 'r2_', 'r1_')),
                             splited[::-1])
            temp = list(temp)[::-1]
            if len(temp) < 3:
                continue

            v = '.'.join(temp)
            last = temp[-1]
            if last.startswith('ci_'):
                candidates.append(v)
                continue
            else:
                if temp[1] == 'co_Russia':
                    if last.startswith('r2_'):
                        candidates.append(v)
                        continue
                elif last.startswith(('r2_', 'r1_')):
                    candidates.append(v)
                    continue

        # если несколько городов указано, говорим что не знаем город
        if len(candidates) == 1:
            return candidates[0]


class TestExtractCity(unittest.TestCase):
    def test_0(self):
        row = [
            1,
            'p_Europe.co_Russia.r1_Urals.r2_YamaloNenetskijAO.ci_NovyUrengoy'
        ]
        self.assertEqual(
            extract_city(row),
            'p_Europe.co_Russia.r1_Urals.r2_YamaloNenetskijAO.ci_NovyUrengoy')

    def test_01(self):
        row = [
            1, 'p_Europe.co_Russia.r1_Urals',
            'p_Europe.co_Russia.r1_Urals.r2_YamaloNenetskijAO'
        ]
        self.assertEqual(extract_city(row),
                         'p_Europe.co_Russia.r1_Urals.r2_YamaloNenetskijAO')

    def test_multiple_cities(self):
        row = [
            1,
            "p_Europe.co_Russia.r1_CentralRussia.r2_Kostroma; p_Europe.co_Russia.r1_CentralRussia.r2_Kursk"
        ]
        self.assertEqual(extract_city(row), None)

    def test_2(self):
        row = [1, "p_Europe.co_Russia.r1_CentralRussia.r2_Kostroma"]
        self.assertEqual(extract_city(row),
                         "p_Europe.co_Russia.r1_CentralRussia.r2_Kostroma")

    def test_3(self):
        row = [1, "p_Europe.co_Russia.r1_CentralRussia"]
        self.assertEqual(extract_city(row), None)

    def test_4(self):
        row = [1, 'p_Europe.co_Belorussia.r1_Brest']
        self.assertEqual(extract_city(row), 'p_Europe.co_Belorussia.r1_Brest')

    def test_5(self):
        row = [
            1,
            'p_Europe.co_Russia.r1_CentralRussia.r2_MosObl.ci_Moscow.cr1_CentralDistrict'
        ]
        self.assertEqual(
            extract_city(row),
            'p_Europe.co_Russia.r1_CentralRussia.r2_MosObl.ci_Moscow')

    def test_6(self):
        row = [1, "p_Europe.co_Russia"]
        self.assertEqual(extract_city(row), None)

    def test_7(self):
        row = [1, "p_Europe"]
        self.assertEqual(extract_city(row), None)

    def test_8(self):
        row = [1, ""]
        self.assertEqual(extract_city(row), None)

    def test_9(self):
        row = [
            1, "p_Europe.co_Russia.r1_CentralRussia",
            'p_Europe.co_Russia.r1_CentralRussia.r2_Kostroma'
        ]
        self.assertEqual(extract_city(row),
                         'p_Europe.co_Russia.r1_CentralRussia.r2_Kostroma')

    def test_10(self):
        row = [
            1,
            'p_Europe.co_Russia.r1_Urals.r2_KhantyMansijskijAO.r3_BeryozovskyDistrict; p_Europe.co_Russia.r1_Urals.r2_Sverdlovsk'
        ]
        self.assertEqual(extract_city(row), None)

    def test_11(self):
        row = [
            1,
            'p_Europe.co_Russia.r1_Urals.r2_KhantyMansijskijAO.r3_BeryozovskyDistrict; p_Europe.co_Russia.r1_Urals'
        ]
        self.assertEqual(extract_city(row),
                         'p_Europe.co_Russia.r1_Urals.r2_KhantyMansijskijAO')

    def test_12(self):
        row = [
            1,
            "p_Europe.co_Ukraine.r1_Zhitomir; p_Europe.co_Ukraine.r1_Dnepropetrovsk"
        ]
        self.assertEqual(extract_city(row), None)

    def test_13(self):
        row = [1, "p_NorthAmerica.co_USA.r1_Ohio"]
        self.assertEqual(extract_city(row), 'p_NorthAmerica.co_USA.r1_Ohio')

    def test_14(self):
        row = [
            1,
            "p_Europe.co_Ukraine.r1_Rovny; p_Europe.co_Russia.r1_CentralRussia.r2_Belgorod; p_Europe.co_Ukraine.r1_Lugansk"
        ]
        self.assertEqual(extract_city(row), None)

    def test_15(self):
        row = [
            1,
            "p_Europe.co_Azerbaijan; p_Europe.co_Russia.r1_SouthOfRussia.r2_Astrakhan"
        ]
        self.assertEqual(extract_city(row),
                         'p_Europe.co_Russia.r1_SouthOfRussia.r2_Astrakhan')

    def test_16(self):
        row = [
            1, "p_Europe.co_Azerbaijan; p_Europe.co_Russia.r1_SouthOfRussia"
        ]
        self.assertEqual(extract_city(row), None)

    def test_17(self):
        row = [
            1,
            "p_Europe.co_Russia.r1_CentralRussia; p_Europe.co_Russia.r1_CentralRussia.r2_Bryansk.ci_Bryansk"
        ]
        self.assertEqual(
            extract_city(row),
            'p_Europe.co_Russia.r1_CentralRussia.r2_Bryansk.ci_Bryansk')

    def test_18(self):
        row = [1, 'p_Asia.co_Vietnam.r1_SouthCentralCoast.r2_KhanhHoa']
        self.assertEqual(extract_city(row),
                         'p_Asia.co_Vietnam.r1_SouthCentralCoast.r2_KhanhHoa')

    def test_19(self):
        row = [
            227613,
            'p_Europe.co_Russia.r1_NorthWest.r2_SpbObl.ci_SaintPetersburg',
            'p_Europe.co_Russia.r1_NorthWest.r2_SpbObl.ci_SaintPetersburg', '',
            '', '', ''
        ]
        self.assertEqual(
            extract_city(row),
            'p_Europe.co_Russia.r1_NorthWest.r2_SpbObl.ci_SaintPetersburg')


if __name__ == "__main__":
    unittest.main()
