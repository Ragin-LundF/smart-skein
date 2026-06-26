# examples

Runnable Skein samples. **Not published.** Depends on all library modules.

## Transaction categorization (classify → route → extract)

`TransactionCategorizationExample` shows the three building blocks composed end to end:

1. **Classify** — a `ClassificationService` (Naive Bayes, privacy-preserving feature hashing)
   learns transaction categories (`insurance` / `rent` / `salary`) from purpose text.
2. **Route** — the predicted category selects category-specific extraction rules.
3. **Extract** — a `SlotExtractor` pulls the structured values that matter for that category
   (insurer/amount pairs for insurance, customer number for rent).

## Run it

```
./gradlew :examples:run
```

Sample output:

```
purpose : AIG-Life 67.89 Geico-Auto 120.00 insurance premium
  category   : insurance (confidence 1.00)
  extracted  : insurer = AIG-Life
  extracted  : amount = 67.89
  extracted  : insurer = Geico-Auto
  extracted  : amount = 120.00
purpose : rent apartment CustomerNumber CD456 monthly
  category   : rent (confidence 1.00)
  extracted  : customer = CD456
```

The end-to-end behavior is asserted in `TransactionCategorizationExampleTest`.
