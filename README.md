# Balkamrai hypertrófia
## Amit érdemes tudni
*Alábbi gyűjtéssel a célom, hogy a területen nem jártas új diákoknak gyors összefoglalót adjak a témával kapcsolatos szakkifejezések jelentéséről, hogy ezáltal is könnyebben fel tudják venni a tempót.

Az esetleges pontatlanságokért elnézést kérek.*

**Balkamrai hypertrófia** *(angol: left ventricular hypertrophy, LVH)*: a bal szívkamra falának kóros megnagyobbodása. Ennek során a szívizomsejtek mérete megnő, de a számuk nem változik, mellyel az ellátó erek nem tudnak lépést tartani, így a szív vérellátása romlik. Oka általában az, hogy a szívnek nagyobb ellenállás (pl. magas vérnyomás esetén) ellen kell dolgoznia. Következményei a szívelégtelenség és a szívinfarktus megnövekedett kockázata.

**Systole**: a szívciklus azon szakasza, mikor a szív összehúzódik, ekkor túlnyomás van a szívben.

**Diastole**: a szívciklus azon szakasza, mikor a szívizmok elernyednek, a szív kitágul.

**Miocardia**: szívizom

**HCM** *(Hypertrophic cardiomyopathy)*: a hypertrófia egyik alfaja, gyakran örökletes

**Legfontosabb diagnózis-típusok**:
- egészséges
- HCM-es
- egyéb hypertrófiás
- sportoló (nekik a megerőltetéstől ugyanúgy megvastagodhat a szívfaluk, azonban ez kevésbé jelent veszélyt)

**MRI** *(Magnetic Resonance Imaging)*: mágneses rezonancia képalkotás, adataink legfőbb forrása. Az emberi test (esetünkben a szív) szeleteiről készít képeket, melyek használhatóak annak térbeli rekonstrukciójára. Ezekkel a képekkel dolgozunk. Ha van rá módunk, érdemes a szív MRI képekből azokat kiválogatni, melyeknél a szív maximálisan ki van tágulva (diastole végi állapotban van), mert a diagnózis ezeken a legkönnyebb.

**MRI típusok**:
- **SA** *(short axis)*: a szív rövid tengelye mentén készült MRI képek, amikhez esetünkben kontúrok is tartoznak, melyek a szív adott részeinek határát jelölik.
- **LA** *(long axis)*: a szív hosszú tengelye mentén készült MRI képek, lehet 2, 3 vagy 4 kamrás, attól függően, hogy milyen szögben készült, és ezáltal hány kamrát tartalmaz.
- **LE** *(late enhancement)*: olyan képek, melyek esetében a páciensben valamilyen kontrasztanyag található, melynek hatására az egyes szövettípusok másképpen reagálnak a mágnesességre, így jobban elkülöníthetőek.
- **TRA**: egyelőre nem használjuk
- Tehát így a lehetséges típusok: SA, SALE, LA (2-3-4 kamrás), LALE (2-3-4 kamrás)

**DICOM fájl**: orvosi fájlformátum, mely esetünkben az MRI képeket tartalmazza, további metaadatok mellett. Ilyen fájlok böngészésére Windows alatt a **MicroDicom** szoftver ajánlott, feldolgozásukra Python nyelven pedig a **Pydicom** könyvtár.

**Adataugmentáció**: mivel orvosi képfeldolgozásban általában jóval kisebb mennyiségű adat (kép) áll rendelkezésünkre, mint hagyományos gépi tanulással végzett képfeldolgozás esetén, ezért szükségünk van arra, hogy az adatmennyiségünket mesterségesen megnöveljük. Ez képek esetén jelenthet például átméretezést, eltolást, tükrözést, forgatást, nyírást (jelen esetben nem ajánlott), zaj hozzáadását, sötétítést-világosítást, vagy gamma-korrekciót; valamilyen intervallumon belül, véletlenszerű paraméterekkel. Így egyrészről növeljük a tanító adatok mennyiségét, másrészről segítjük hogy a háló ezekre a transzformációkra invariánssá váljon, végül pedig nehezebbé tesszük, hogy a háló túltanulja (*overfitting*) magát. Tehát ezen a területen érdemes adataugmentációt használni.

## Elvégzett munka
### Adat előfeldolgozás
A DICOM fájlok meta tag-jeinek kényelmes, szűrhető vizsgálatához segédosztályt készítettem (**DicomFilter.py**). Az osztály adott mappákban, vagy akár rekurzívan ezek almappáiban is beolvassa a DICOM fájlokat, és azokból egy *Pandas dataframe*-et készít, mely jóval kényelmesebben elemezhető, mind személyesen, mind algoritmikusan. Elsődlegesen azokat az attribútumokat tartja meg, melyek nem minden beolvasott fájl esetén azonosak. Így segítséget nyújt egy adott esetben lényeges DICOM attribútumokról és azok értékeiről. Tovább megadhatók neki attribútumok egy listában (*keepAttributes*), melyeket semmiképpen sem szűr ki, és olyanok is (*dropAttributes*), melyeket mindenképpen kiszűr.

Elkészítettem egy adatbeolvasó és -feldolgozó programot (**data_reader.py**), mely a különböző kategóriájú MRI képeket külön lekezelve, a hiányokat, váratlan eseteket is jól tűrve az egyes páciensekhez tartozó adatokat beolvassa, szűri, és azokat egy bináris fájlba menti a **Pickle** könyvtár segítségével. Az egyes MRI-típusok esetén a következőkre kellett figyelni:
- **SA**: A **dicom_reader.py** és **con_reader.py** osztályok csak ehhez a típushoz használhatóak. Segítségükkel megfelelő sorrendben beolvashatók az egyes képsorozatok, és azokból a contúrok területének számításával kiválaszthatóak a maximális térfogatokhoz tartozó képek.
- **SALE**: A *sliceLocation* tag-re kellett figyelni, az elmentendő sorozat addig a kép előttig tart, ahol a *sliceLocation* újra nullától kezdődik.
- **LA**: Itt is a *sliceLocation* tag-et volt érdemes figyelni, minden egyes ugrásakor egy másik kamra-nézetbe kerültünk, mely sorozatokból feltételezni lehetett, hogy az elsőkhöz tartozik a maximális térfogat, így ezeket mentettem el a következő sorrendben: 2 kamra, 4 kamra, 3 kamra. (Vigyázat, a technika a teszt adatokon jól működött, azonban az teljes adathalmazon további ellenőrzést igényel.)
- **LALE**: Az itt található képeknek sorrendben pontosan a középső harmadát mentettem el.

### Tanítás számos paraméter vizsgálatával
Az elkészült előfeldolgozás után tanításokat végeztem (**Trainer.py**) az adatokon, modellek, technikák, és MRI-típusok széles skáláját felhasználva. Ennek célja az volt, hogy felmérjem, melyik irányokba érdemes a leginkább elindulni, mik azok, amik jól működnek, mik azok, amik nem. Kifejezetten nem egyetlen módszer maximális optimalizálása volt a cél, hanem minél több felderítése. A tanítások eredményeit és a tanulságokat a következő fejezetben taglalom.

## Eredmények
A referencia módszerem, melyhez képest a egyes paramétereket változtattam, a következő volt:
ResNet18 modell, SA MRI típusú képeken, 3 diagnózis-osztályon, 32-es batch mérettel, közepes adataugmentációval 100 epochig tanult. Ezeknek a paramétereknek egyikét változtattam a későbbiekben.

### A különböző MRI típusok eredményei
Teszteltem a fent leírt modellt az összes lehetséges MRI típuson. A grafikon alapján elmondható, hogy általánosságban az SA típusú képeken szignifikánsan jobban teljesít az algoritmus, míg az LE típusú képek mindkét fő MRI típus esetében enyhe eredményjavuláshoz (vagyis loss-csökkenéshez vezettek).
![A különböző MRI típusok eredményei](https://drive.google.com/uc?id=1J_I68Emaovwy_hPA4spbtwKia4eONsXK)

### Tesztelt architektúrák eredményei
4 különböző modellen végeztem tanítást: VGG11 (batch norm), MobileNet v2, ResNet18, SqueezeNet 1.0. A modellek kiválasztásakor ügyeltem arra, hogy minél kevesebb paramétert tartalmazzanak, hiszen viszonylag kevés adatunk van, és ügyelni kell arra, hogy a túltanítást elkerüljük. Az eredmények alapján elmondható, hogy általánosságban a VGG modell vezetett a legnagyobb pontossághoz, azonban ha a tanítás idejét is figyelembe vesszük, a ResNet mindkét téren jól teljesített, és egy jó kompromisszumnak tűnik a tanítási idő és pontosság között.
![Tesztelt architektúrák eredményei](https://drive.google.com/uc?id=1PH1IXDVlzogiw6ux3fsFJiQrRe6SQdpR)

### Diagnózis-típusok darabszámának hatása
Az osztályok számának hatása egy érdekes eredményt mutat: a tanítás nagyobb pontosságra vezetett 4 osztály, mint 3 osztály esetén. Tehát a jövőbeli tanítások esetén ezzel érdemes lehet kísérletezni.
![Diagnózis-típusok darabszámának hatása](https://drive.google.com/uc?id=1rwVJWd8SELf58aX9uo-neXOUipA5niVD)

### Augmentáció erősségének hatása
Különböző augmentációkat is kipróbáltam:
- *minimális*: 2x nagyítás, 224x224 CenterCrop, semmi véletlenszerűség
- *normális*: 1.8-2.2x nagyítás, +-0.1x képszélességnyi eltolás, és +-20 foknyi forgatás
- *maximális*: 1.7-2.3x nagyítás, +-0.15x képszélességnyi eltolás, +-30 foknyi forgatás és ColorJitter 0.3-as brightness-el

Rövid- és közép távon a normális adataugmentáció mutatta a legjobb eredményeket, azonban hosszú távon a maximális adataugmentáció esetén pontosabbá vált a modell, tehát a jövőben egy jelentősebb adataugmentációt érdemes lehet megfontolni.
![Augmentáció erősségének hatása](https://drive.google.com/uc?id=1NX08cheZ7MQrIpaSaYflzWx03xZ3yA8F)

### Transzfer tanítás hatása
Kísérleteztem transzfertanítással is. Itt a három tesztelt technika a következő:
- *retrain*: hagyományos tanítás, véletlen inicializálással
- *finetune*: ImageNet-en betanított súlyokkal betöltött modell tanítása az új adatokon
- *transfer*: ImageNet-en betanított súlyokkal betöltött modell utolsó, klasszifikáló rétegének tanítása az új adatokon

A *transfer* technika kezdettől fogva nem vezetett nagy pontosságra, ami az ImageNet és a mi adathalmazunk különbözőségét megfigyelve nem is meglepő. Ami meglepő volt azonban, az viszont az, hogy a *finetune* modell már a tanítás elején nagyon jó pontosságra volt képes, és még hosszú távon is enyhén jobb eredményre volt képes, mint a hagyományosan tanított modell.
![Transzfer tanítás hatása](https://drive.google.com/uc?id=1S0lVBqgr3I_aTN2-XE7Xg9BdzVn-fd0Q)

### Igazságmátrix 2, 3, 4 diagnózis-típus esetén
Végezetül pedig ábrázoltam a különböző osztály-felosztások esetén létrejött igazságmátrixokat. Ezek optimális esetben egységmátrixok lennének (valamilyen skálázással). Azt láthatjuk, hogy mindhárom esetben jó pontosságot ért el a modell, nem látszanak kiugró egyensúlytalanságok. Külön kiemelném, hogy a 4 osztály esetében a modell milyen pontossággal különbözteti meg a sportolók, HCM-esek és egyéb hypertrófiások képeit, annak ellenére, hogy mindhárom osztály esetében a szív megvastagszik, tehát pusztán erre nem hagyatkozhat.
![Igazságmátrix 2 diagnózis-típus esetén](https://drive.google.com/uc?id=1qe7KHs09CTciAe9_BcAHO8X_6JCEm2jI)
![Igazságmátrix 3 diagnózis-típus esetén](https://drive.google.com/uc?id=17YijzBWUhsl7yXufsc2egqRe4XVTTSkI)
![Igazságmátrix 4 diagnózis-típus esetén](https://drive.google.com/uc?id=12JoJ7IEuWFYxznqRgAjjbMk7G54ZbVyc)


*Béres András*
