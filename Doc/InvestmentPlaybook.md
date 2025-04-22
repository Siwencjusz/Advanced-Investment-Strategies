# IT‑Tata Final Investment Playbook

> **Wersja Ostateczna** – krok po kroku, jak dla małpy, bez miejsca na wymysły.

## 🔑 Kluczowe zasady
1. **Nigdy nie tracisz >20%** (ogólny portfel).  
2. **Stosujesz się do instrukcji bez wyjątków**.  
3. **Zero kombinacji** = zero strat.  
4. **Dokładny zapis** każdej decyzji.

---

## 🗓 Dzienna Rutyna (5 min)
1. **Otwórz komputer**.  
2. **Zaloguj się** do brokera (XTB/IBKR/BOSSA). **Max 3 logowania dziennie**.  
3. **Sprawdź cash buffer**:  
   - < 30 000 zł? → kup obligacje EDO.  
4. **Gotówka > 60 000 zł?** → wystaw 1–2 opcje CSP na VWRA (strike = -5%).  
5. **Wyloguj się** i **zamknij** stronę brokera.

---

## 📆 Roczne Wpłaty (styczeń)
- **IKE (Super IKE)**: wpłacasz max 23 472 zł → kupujesz ETF V80A lub VWRA.  
- **IKZE**: wpłacasz max 9 120 zł → kupujesz obligacje EDO/ROD.  
- **OIPE**: wpłacasz 1 000 EUR (ok. 5 000 zł) → kupujesz globalny ETF (VWRA).

> **Alternatywa 50/50:**  
> Jeśli boisz się wejścia na szczycie, połowę wpłacasz w styczniu, połowę w lipcu.

---

## 📅 Miesięczne Wpłaty (1., 10., 20.)

### 1. Dodatkowe wpłaty
- **5 000 zł** → rachunek maklerski (XTB).  
- **1 000 zł** (ok. 300 EUR) → OIPE.  
- **2 000 zł** → obligacje EDO.

### 2. Core DCA (bez SL)
- **Kupujesz ETF** (V80A/VWRA) za 1 000 zł na XTB.

### 3. Satellite (z SL)
- **Sprawdź GEM/TAA** w aplikacji.  
- **GEM ON?** → kup 1 000 zł ETF momentum + ustaw SL.  
- **TAA ON?** → kup 1 000 zł ETF makro + ustaw SL.
- **Brak sygnału?** → nic nie robisz.

### 4. Crypto AEM (z SL)
- **BTC >200 SMA?** → kup 500 zł crypto ETF + ustaw SL.  
- **BTC <200 SMA?** → kup 500 zł T‑Bills/EDO.

### 5. Boost (opcje)
- **Gotówka >60 000 zł** i nie więcej niż 2 CSP (strike -5%).  
- **Trend boczny ETF?** → 2 Covered Calls (strike +5%).

---

## ✅ Pre‑Trade Checklist (5 pkt)
1. **Edge:** czemu to kupujesz?  
2. **Sizing:** ≤ 1% wartości portfela.  
3. **Signal ON:** GEM/TAA lub BTC>200 SMA.  
4. **SL ustawiony:** zgodnie z poniższą instrukcją.  
5. **Plan B:** gap-down? co zrobię?

> Jeśli nie 5/5 → **STOP**.

---

## 🚫 Ustawianie Stop Loss (SL)
1. **Sprawdź cenę** ETF (np. 100 zł).  
2. **Otwórz TradingView**, wpisz symbol (np. VWRA).  
3. **Dodaj wskaźnik ATR (14 dni)** → odczytaj wartość (np. ATR=3 zł).  
4. **Rachunek:** SL = cena - (2×ATR).  
   - Przykład: 100 zł - (2×3 zł) = 94 zł.  
5. **Wpisz SL** u brokera przy składaniu zlecenia.

---

## 🚨 Zdarzenia awaryjne
- **Satellite -15% DD** → pauza Satellite do sygnału ON.  
- **Portfel -25% DD** → sprzedajesz wszystko, pauza 30 dni.

---

## 🛡️ Hedging
- **Trigger:** VIX>25 lub MOVE>150.  
- **Kupujesz** 10% portfela w VIXM lub PUT_SPREAD_ACWI.

---

## 💰 Zarządzanie gotówką

- **Cushion:** 6 m-cy wydatków (~60 000 zł) w EDO/ROD.  
- **Min cash:** 30 000 zł → refill EDO.  
- **Max cash:** 60 000 zł → CSP.

---

## 🔧 Last‑Week Adjustment (niwelowanie "dziury")

*Parametry:*  
```text
total_monthly:      5000 PLN
weekly_base:        1250 PLN
Strategie alokacji:
  TAA:   45% → 562.50 PLN (~150 USD / 600 USD)
  GEM:   27.5% → 343.75 PLN (~80 EUR / 320 EUR)
  DCA:   25% → 312.50 PLN (~75 EUR / 300 EUR)
  AEM:   2.5% → 31.25 PLN (~15 EUR / 60 EUR)
Dni_offset: [0,7,14,21,28]  # → zakupy 20., 27., 6./7., 13./14., 20./21. dnia
Godzina: 11:00
```

*Problem:* 4. tydzień może mieć nadwyżkę, bo AEM wymaga min 15 EUR (~63 PLN).

1. **Schemat zakupów:**
   ```python
   # import datetime, scheduler etc.
   total_monthly = 5000
   weekly_base   = 1250
   allocations = {
       'TAA': 0.45,
       'GEM': 0.275,
       'DCA': 0.25,
       'AEM': 0.025
   }
   aem_min_pln = 63
   offsets = [0,7,14,21,28]

   for i, offset in enumerate(offsets):
       # harmonogram: dzień 20. + offset dni
       run_date = calculate_date(base_day=20, offset_days=offset, time='11:00')

       if i < 4:
           kwota = weekly_base
       else:
           spent = weekly_base * 4  # 4 tygodnie bazowe
           remaining = total_monthly - (weekly_base * 3)
           # uwzględnij min AEM:
           required_sum = sum([remaining * allocations[s] for s in ['TAA','GEM','DCA']]) + aem_min_pln
           kwota = max(remaining, required_sum)

       # wykonanie zakupów
       buy_TAA(amount=kwota * allocations['TAA'], schedule=run_date)
       buy_GEM(amount=kwota * allocations['GEM'], schedule=run_date)
       buy_DCA(amount=kwota * allocations['DCA'], schedule=run_date)
       buy_AEM(amount=min(kwota * allocations['AEM'], aem_min_pln), schedule=run_date)
   ```

2. **Zasady KISS:**
   - Tygodnie 1–4: kupujesz według `weekly_base = 1250 PLN` i proporcji.
   - Dni_offset i godzina → precyzyjny harmonogram 20/27/6/13/20 dnia miesiąca o 11:00.

3. **PlanRollback:**
   - Jeśli `aem_min_pln > remaining`: redukuj alokacje lub przelej nadwyżkę na cash.
   - Zaokrąglenia: kompensuj różnice następnym miesiącu.
   - `API_FAIL`: retry następnego dnia o 11:00 lub lump-sum za miesiąc.

---

## 🎰 Konto hazardowe (Alior)

- **Cel:** 100–300 zł miesięcznie na małe spekulacje.
- **Automatyzacja:** ustaw stały przelew 100 zł 5. dnia miesiąca.
- **Zasada:** tylko z tego konta możesz robić "wolne wejścia" bez reguł SL/TP.

---

## 🐵 Złote zasady małpy
- Nie myśl.  
- Nie kombinuj.  
- Rób dokładnie to, co tu jest napisane.

**Gotowy snippet Python+APScheduler lub inna konfiguracja? Napisz "TAK snippet" lub "zmień liczbę zakupów".**

