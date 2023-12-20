import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from lm_transformer.repos.lm_transformer_repo import create_lm_transformer_detail


# push to queue for DB updates
def save_predicted_answer(predicted_answer, question_asked, question_context):
    resp = create_lm_transformer_detail(predicted_answer, question_asked, question_context)
    return True if resp else False

# Step 1: Load Pretrained Model and Tokenizer
# Step 2: Prepare training dataset
# Step 3: Prep Training Arguments
# training_args = TrainingArguments(
#     output_dir="./custom_model",  # Directory to save the fine-tuned model
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     save_steps=5000,  # Save model every 5000 steps
#     save_total_limit=2,  # Only keep the last 2 saved models
# )
# Step 4: Fine-Tune the Model
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
# )
# trainer.train()
# Step 5: Save the Fine-Tuned Model Locally
# model.save_pretrained("./custom_model")

def load_fine_tuned_model_and_tokenizer():
    # Load the model
    model_checkpoint = "/Users/shwetasingh/Projects/Project-Berry/berry_information_engine/lm_transformer/fine_tuned_lms/checkpoint-33276"
    model_tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    fine_tuned_model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    return fine_tuned_model, model_tokenizer


def predict_answer(fine_tuned_model, model_tokenizer, question_asked, question_context):
    # Use the model for predictions
    input_encoding = model_tokenizer(question_context, question_asked, return_tensors="pt")
    with torch.no_grad():
        outputs = fine_tuned_model(**input_encoding)

    # Print the prediction
    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = input_encoding.input_ids[0, answer_start_index: answer_end_index + 1]
    predicted_answer = model_tokenizer.decode(predict_answer_tokens)

    print(f"Predicted Answer: {predicted_answer}")
    return predicted_answer


def get_predicted_answer_for_question_and_context(question_context, question_asked):
    # question_asked = "How many programming languages does BLOOM support?"
    # question_context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
    predicted_answer = ''
    fine_tuned_model, model_tokenizer = load_fine_tuned_model_and_tokenizer()
    predicted_answer = predict_answer(fine_tuned_model, model_tokenizer, question_asked, question_context)
    save_predicted_answer(predicted_answer, question_asked, question_context)
    return predicted_answer, question_context, question_asked

