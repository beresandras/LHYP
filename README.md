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
 - **SA** *(short axis)*: a szív rövid tengelye mentén készült MRI képek
 - **LA** *(long axis)*: a szív hosszú tengelye mentén készült MRI képek, lehet 2, 3 vagy 4 kamrás, attól függően, hogy milyen szögben készült, és ezáltal hány kamrát tartalmaz.
 - **LE** *(late enhancement)*: olyan képek, melyek esetében a páciensben valamilyen kontrasztanyag található, melynek hatására az egyes szövettípusok másképpen reagálnak a mágnesességre, így jobban elkülöníthetőek.
 - Tehát így a lehetséges típusok: SA, SALE, LA (2-3-4 kamrás), LALE (2-3-4 kamrás)
**DICOM fájl**: orvosi fájlformátum, mely esetünkben az MRI képeket tartalmazza, további metaadatok mellett. Ilyen fájlok böngészésére Windows alatt a **MicroDicom** szoftver ajánlott, feldolgozásukra Python nyelven pedig a **Pydicom** könyvtár.
**Adataugmentáció**: mivel orvosi képfeldolgozásban általában jóval kisebb mennyiségű adat (kép) áll rendelkezésünkre, mint hagyományos gépi tanulással végzett képfeldolgozás esetén, ezért szükségünk van arra, hogy az adatmennyiségünket mesterségesen megnöveljük. Ez képek esetén jelenthet például átméretezést, eltolást, tükrözést, forgatást, nyírást (jelen esetben nem ajánlott), zaj hozzáadását, sötétítést-világosítást, vagy gamma-korrekciót; valamilyen intervallumon belül, véletlenszerű paraméterekkel. Így egyrészről növeljük a tanító adatok mennyiségét, másrészről segítjük hogy a háló ezekre a transzformációkra invariánssá váljon, végül pedig nehezebbé tesszük, hogy a háló túltanulja (*overfitting*) magát. Tehát ezen a területen érdemes adataugmentációt használni.

## Elvégzett munka
### Adat előfeldolgozás
A DICOM fájlok meta tag-jeinek kényelmes, szűrhető vizsgálatához segédosztályt készítettem (**DicomFilter**). Az osztály adott mappákban, vagy akár rekurzívan ezek almappáiban is beolvassa a DICOM fájlokat, és azokból egy *Pandas dataframe*-et készít, mely jóval kényelmesebben elemezhető, mind személyesen, mind algoritmikusan. Elsődlegesen azokat az attribútumokat tartja meg, melyek nem minden beolvasott fájl esetén azonosak. Így segítséget nyújt egy adott esetben lényeges DICOM attribútumokról és azok értékeiről. Tovább megadhatók neki attribútumok egy listában (*keepAttributes*), melyeket semmiképpen sem szűr ki, és olyanok is (*dropAttributes*), melyeket mindenképpen kiszűr.

### Tanítás számos paraméter vizsgálatával
## Eredmények
![MRI típusok](https://raw.githubusercontent.com/beresandras/LHYP/beresandras/charts/Different_models.png)
![Tesztelt architektúrák](https://raw.githubusercontent.com/beresandras/LHYP/beresandras/charts/Different_models.png)
