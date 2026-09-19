package io.skein.examples.recipes

/**
 * The two corpora this example turns on.
 *
 * **Training** is generated from a seeded combinatorial template — cuisine x protein x method x
 * time — and labelled by running the ruleset. It is committed as a generator with a fixed seed
 * rather than as data, so it is reproducible and readable.
 *
 * **Held-out** is forty hand-written recipes using wording the rules *miss*: "aubergine" where the
 * rules say nothing about it, "courgette" for "zucchini", "cacio e pepe" with no `pasta` in the
 * title, "sans gluten". They are labelled by hand, the way a person would.
 *
 * That second set is the entire point. A model trained only on rule-generated data largely learns
 * the rules, and scoring it against more rule-generated data measures how well it reproduces them.
 * The hand-written set is the only thing here that answers the question a team replacing a rule
 * engine actually has — *does it generalise past the rules?* — and an example that leaves it out
 * teaches the wrong lesson.
 */
object RecipeCorpus {

    private data class Cuisine(val name: String, val titles: List<String>, val markers: List<String>)

    private val CUISINES = listOf(
        Cuisine(
            name = "italian",
            titles = listOf("pasta bake", "risotto", "pizza", "lasagne"),
            markers = listOf("parmesan", "basil", "pancetta"),
        ),
        Cuisine(
            name = "mexican",
            titles = listOf("taco bowl", "burrito", "quesadilla"),
            markers = listOf("tortilla", "jalapeno", "chipotle"),
        ),
        Cuisine(
            name = "plain",
            titles = listOf("traybake", "stew", "salad bowl", "skillet supper"),
            markers = listOf("onion", "carrot", "thyme"),
        ),
        Cuisine(
            name = "sweet",
            titles = listOf("chocolate cake", "apple tart", "brownie", "rice pudding"),
            markers = listOf("sugar", "vanilla", "cocoa"),
        ),
    )

    private val PROTEINS = listOf(
        "beef", "chicken", "pork", "fish", "bacon",
        "tofu", "beans", "mushroom", "lentils", "chickpeas",
    )

    /** Recipes generated per template, so a template is a group worth keeping whole. */
    private const val VARIANTS_PER_TEMPLATE = 2

    private val DAIRY = listOf("butter", "cheese", "cream", "milk", "egg")
    private val STARCHES = listOf("flour", "pasta", "bread", "rice", "potato", "quinoa")
    private val METHODS = listOf("one pot", "baked", "grilled", "fried", "simmered")
    private val TIMES = listOf(10, 15, 20, 30, 45, 60)

    /**
     * The generated training corpus.
     *
     * A row's [Recipe.origin] is the template it came from, which is the honest grouping key here:
     * every row sharing a template shares most of its wording, exactly as every example of one rule
     * shared that rule's keyword in the original workflow.
     */
    fun training(): List<Recipe> {
        val recipes = ArrayList<Recipe>()
        var step = 0
        CUISINES.forEach { cuisine ->
            PROTEINS.forEach { protein ->
                METHODS.forEach { method ->
                    val template = "${cuisine.name}/$protein/$method"
                    // Several recipes per template, so a template is a group with siblings in it.
                    // One recipe per template would make every group a singleton and grouped
                    // cross-validation identical to an ungrouped one -- the leak this example is
                    // built to show would be invisible.
                    repeat(times = VARIANTS_PER_TEMPLATE) { variant ->
                        // A seeded, fully deterministic walk through the remaining axes: no Random,
                        // so the corpus is identical on every machine and every run.
                        val turn = step + variant
                        val dairy = if (turn % 3 == 0) DAIRY[turn % DAIRY.size] else null
                        recipes.add(
                            element = Recipe(
                                title = "$protein ${cuisine.titles[turn % cuisine.titles.size]}",
                                ingredients = listOfNotNull(
                                    protein,
                                    cuisine.markers[turn % cuisine.markers.size],
                                    STARCHES[turn % STARCHES.size],
                                    dairy,
                                ).joinToString(separator = " "),
                                method = method,
                                minutes = TIMES[turn % TIMES.size],
                                origin = template,
                            ),
                        )
                    }
                    step += VARIANTS_PER_TEMPLATE
                }
            }
        }
        return recipes
    }

    /**
     * Forty hand-written recipes whose wording the ruleset does not cover, each labelled by hand.
     *
     * Expect these to score substantially worse than the generated holdout. That gap is the
     * measurement, not a defect.
     */
    // A committed data table, not logic: the length is forty hand-written rows and the "magic
    // numbers" are cooking times. Extracting either into constants would obscure the data without
    // making anything testable.
    @Suppress("LongMethod", "MagicNumber")
    fun handWritten(): List<Pair<Recipe, Set<String>>> {
        return listOf(
            hand("aubergine parmigiana", "aubergine parmesan basil tomato", "baked", 55,
                "VEGETARIAN", "ITALIAN"),
            hand("cacio e pepe", "pecorino black pepper spaghetti", "simmered", 15,
                "VEGETARIAN", "ITALIAN", "QUICK"),
            hand("courgette fritters", "courgette egg flour dill", "fried", 18,
                "VEGETARIAN", "QUICK"),
            hand("sans gluten almond sponge", "ground almonds egg sugar", "baked", 40,
                "VEGETARIAN", "GLUTEN_FREE", "DESSERT"),
            hand("chana masala", "chickpeas tomato ginger garam masala", "one pot", 35,
                "VEGETARIAN", "VEGAN", "GLUTEN_FREE", "ONE_POT"),
            hand("ribollita", "cannellini kale stale sourdough", "one pot", 50,
                "VEGETARIAN", "VEGAN", "ITALIAN", "ONE_POT"),
            hand("elote salad", "sweetcorn lime mayonnaise ancho", "grilled", 20,
                "VEGETARIAN", "MEXICAN", "GLUTEN_FREE", "QUICK"),
            hand("pollo asado", "chicken thighs lime ancho oregano", "grilled", 45,
                "MEXICAN"),
            hand("bolognese ragu", "minced beef soffritto red wine", "simmered", 90,
                "ITALIAN"),
            hand("pan con tomate", "sourdough ripe tomato garlic", "grilled", 10,
                "VEGETARIAN", "VEGAN", "QUICK"),
            hand("shakshuka", "eggs peppers cumin tomato", "one pot", 30,
                "VEGETARIAN", "GLUTEN_FREE", "ONE_POT"),
            hand("dal tadka", "red lentils turmeric ghee cumin", "one pot", 40,
                "VEGETARIAN", "GLUTEN_FREE", "ONE_POT"),
            hand("gazpacho", "ripe tomato cucumber pepper olive oil", "simmered", 15,
                "VEGETARIAN", "VEGAN", "GLUTEN_FREE", "QUICK"),
            hand("tiramisu", "mascarpone espresso savoiardi cocoa", "baked", 30,
                "VEGETARIAN", "ITALIAN", "DESSERT"),
            hand("affogato", "vanilla gelato hot espresso", "simmered", 5,
                "VEGETARIAN", "ITALIAN", "GLUTEN_FREE", "DESSERT", "QUICK"),
            hand("carnitas", "pork shoulder orange cumin", "simmered", 180,
                "MEXICAN"),
            hand("esquites cup", "sweetcorn cotija chilli lime", "fried", 15,
                "VEGETARIAN", "MEXICAN", "GLUTEN_FREE", "QUICK"),
            hand("caponata", "aubergine celery capers vinegar", "simmered", 45,
                "VEGETARIAN", "VEGAN", "ITALIAN", "GLUTEN_FREE"),
            hand("panzanella", "stale bread ripe tomato basil", "grilled", 20,
                "VEGETARIAN", "VEGAN", "ITALIAN", "QUICK"),
            hand("arroz con leche", "rice milk cinnamon sugar", "simmered", 40,
                "VEGETARIAN", "MEXICAN", "GLUTEN_FREE", "DESSERT"),
            hand("polenta e funghi", "polenta porcini parmesan", "simmered", 35,
                "VEGETARIAN", "ITALIAN", "GLUTEN_FREE"),
            hand("frijoles de olla", "black beans epazote onion", "one pot", 90,
                "VEGETARIAN", "VEGAN", "MEXICAN", "GLUTEN_FREE", "ONE_POT"),
            hand("pesto alla genovese", "basil pine nuts pecorino", "simmered", 12,
                "VEGETARIAN", "ITALIAN", "GLUTEN_FREE", "QUICK"),
            hand("tinga de pollo", "chicken chipotle adobo onion", "one pot", 40,
                "MEXICAN", "ONE_POT"),
            hand("zucchini carpaccio", "zucchini lemon olive oil almond", "grilled", 12,
                "VEGETARIAN", "VEGAN", "ITALIAN", "GLUTEN_FREE", "QUICK"),
            hand("pico de gallo", "tomato onion coriander lime", "simmered", 10,
                "VEGETARIAN", "VEGAN", "MEXICAN", "GLUTEN_FREE", "QUICK"),
            hand("sopa de lima", "chicken lime tortilla strips", "one pot", 35,
                "MEXICAN", "ONE_POT"),
            hand("melanzane involtini", "aubergine ricotta basil", "baked", 50,
                "VEGETARIAN", "ITALIAN", "GLUTEN_FREE"),
            hand("budino al cioccolato", "dark chocolate cream egg yolk", "baked", 25,
                "VEGETARIAN", "ITALIAN", "GLUTEN_FREE", "DESSERT"),
            hand("nopales salad", "cactus paddles tomato onion", "grilled", 18,
                "VEGETARIAN", "VEGAN", "MEXICAN", "GLUTEN_FREE", "QUICK"),
            hand("minestrone", "borlotti beans soffritto ditalini", "one pot", 55,
                "VEGETARIAN", "ITALIAN", "ONE_POT"),
            hand("guacamole", "avocado lime coriander onion", "simmered", 8,
                "VEGETARIAN", "VEGAN", "MEXICAN", "GLUTEN_FREE", "QUICK"),
            hand("saltimbocca", "veal escalope prosciutto sage", "fried", 20,
                "ITALIAN", "QUICK"),
            hand("calabacitas", "zucchini sweetcorn queso fresco", "fried", 20,
                "VEGETARIAN", "MEXICAN", "GLUTEN_FREE", "QUICK"),
            hand("torta caprese", "ground almonds dark chocolate egg", "baked", 45,
                "VEGETARIAN", "ITALIAN", "GLUTEN_FREE", "DESSERT"),
            hand("albondigas soup", "minced beef mint broth", "one pot", 50,
                "MEXICAN", "ONE_POT"),
            hand("farinata", "chickpea flour rosemary olive oil", "baked", 30,
                "VEGETARIAN", "VEGAN", "ITALIAN", "GLUTEN_FREE"),
            hand("rajas con crema", "poblano strips cream onion", "fried", 25,
                "VEGETARIAN", "MEXICAN", "GLUTEN_FREE"),
            hand("bruschetta", "sourdough ripe tomato basil garlic", "grilled", 10,
                "VEGETARIAN", "VEGAN", "ITALIAN", "QUICK"),
            hand("flan napolitano", "condensed milk egg vanilla", "baked", 60,
                "VEGETARIAN", "MEXICAN", "GLUTEN_FREE", "DESSERT"),
        )
    }

    private fun hand(
        title: String,
        ingredients: String,
        method: String,
        minutes: Int,
        vararg labels: String,
    ): Pair<Recipe, Set<String>> {
        return Recipe(
            title = title,
            ingredients = ingredients,
            method = method,
            minutes = minutes,
            origin = "hand",
        ) to labels.toSet()
    }
}
