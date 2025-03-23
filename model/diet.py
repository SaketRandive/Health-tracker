# models/diet_model.py
import random

def get_diet_recommendation(goal):
    diet_plans = {
        "weight_loss": [
            "Breakfast: Greek yogurt with berries and chia seeds | Lunch: Quinoa salad with grilled chicken and mixed vegetables | Dinner: Baked salmon with steamed asparagus and lemon",
            "Breakfast: Spinach and mushroom egg white omelet | Lunch: Lentil soup with side salad | Dinner: Turkey lettuce wraps with avocado",
            "Breakfast: Green smoothie with spinach, apple, and protein powder | Lunch: Tuna salad on cucumber slices | Dinner: Zucchini noodles with turkey meatballs",
            "Breakfast: Cottage cheese with sliced peaches | Lunch: Grilled chicken with roasted vegetables | Dinner: Baked white fish with sautéed kale",
            "Breakfast: Overnight oats with almond milk and cinnamon | Lunch: Large mixed green salad with grilled tofu | Dinner: Cauliflower rice stir-fry with shrimp"
        ],
        
        "muscle_gain": [
            "Breakfast: Protein pancakes with banana and honey | Lunch: Chicken, brown rice, and broccoli | Dinner: Grass-fed beef steak with sweet potato and green beans",
            "Breakfast: Scrambled eggs with turkey bacon and avocado | Lunch: Tuna sandwich on whole grain bread | Dinner: Salmon with quinoa and roasted vegetables",
            "Breakfast: Oatmeal with whey protein, almond butter and blueberries | Lunch: Lean ground beef with whole wheat pasta | Dinner: Grilled chicken with rice and vegetables",
            "Breakfast: Greek yogurt with granola, nuts and honey | Lunch: Chicken and black bean burrito bowl | Dinner: Turkey meatloaf with mashed potatoes and green peas",
            "Breakfast: Protein shake with banana, peanut butter and oats | Lunch: Grilled chicken sandwich with hummus | Dinner: Baked cod with wild rice and asparagus"
        ],
        
        "balanced": [
            "Breakfast: Whole grain toast with avocado and poached egg | Lunch: Mediterranean bowl with falafel and tahini | Dinner: Grilled chicken with roasted vegetables and quinoa",
            "Breakfast: Oatmeal with mixed fruits and nuts | Lunch: Chickpea and vegetable wrap | Dinner: Baked salmon with sweet potato and green beans",
            "Breakfast: Smoothie bowl with berries, banana and granola | Lunch: Lentil soup with whole grain bread | Dinner: Stir-fried tofu with brown rice and vegetables",
            "Breakfast: Chia seed pudding with mango and coconut | Lunch: Turkey and avocado wrap with hummus | Dinner: Baked fish with roasted Brussels sprouts and wild rice",
            "Breakfast: Cottage cheese with pineapple and walnuts | Lunch: Quinoa bowl with roasted vegetables and feta | Dinner: Grilled chicken with pesto pasta and cherry tomatoes"
        ]
    }
    return random.choice(diet_plans.get(goal, ["Breakfast: Fruit salad with Greek yogurt | Lunch: Mixed green salad with grilled chicken | Dinner: Vegetable soup with whole grain bread"]))
# import google.generativeai as genai
# import os
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# def get_diet_recommendation(goal):
#     """ Get diet recommendation using Gemini AI """
#     model = genai.GenerativeModel("gemini-1.5-flash")
#     prompt = f"Provide a healthy {goal} diet plan for an active person."
#     response = model.generate_content(prompt)
#     return response.text.strip()
