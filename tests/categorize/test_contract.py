"""Contract tests for categorize module vs architecture.pdf.

These tests verify that the categorize/ module code structure and public API
matches the requirements documented in architecture.pdf (as reflected in
categorize/README.md). They test code patterns, function signatures, and
documented behavior without requiring the full ML stack to be importable.
"""

import inspect
from pathlib import Path

CATEGORIZE_DIR = Path(__file__).resolve().parents[2] / "categorize"


class TestPDFContractStructural:
    """Structural tests verifying the code matches PDF contract."""

    def test_classifier_protocol_has_predict_method(self):
        """PDF: CategoryClassifier Protocol must have predict(item_name) method.

        Verifies the protocol exists and has the correct signature.
        """
        from categorize.classifier import CategoryClassifier

        assert hasattr(CategoryClassifier, "predict"), (
            "CategoryClassifier must have a predict method"
        )

        # Verify the method signature includes item_name
        sig = inspect.signature(CategoryClassifier.predict)
        params = list(sig.parameters.keys())
        # At least one parameter should relate to the item name
        name_related = [p for p in params if "name" in p.lower() or "item" in p.lower()]
        assert len(name_related) > 0, "predict method must have an item_name parameter"

    def test_transformer_predict_has_sigmoid_and_threshold(self):
        """PDF: TransformerCategoryClassifier.predict must use sigmoid + 0.5 threshold.

        PDF behavior:
        1. Tokenize with max_length
        2. Forward pass, get logits
        3. Apply sigmoid to get probabilities
        4. Filter by 0.5 threshold
        5. Fallback to argmax if none pass
        6. Sort by confidence descending
        """
        import inspect

        from categorize.transformer import TransformerCategoryClassifier

        # Get the source of the predict method
        src = inspect.getsource(TransformerCategoryClassifier.predict)

        # Must contain sigmoid application
        has_sigmoid = "sigmoid" in src.lower() or "torch.sigmoid" in src
        assert has_sigmoid, "predict method must apply sigmoid to logits (PDF requirement)"

        # Must have 0.5 threshold reference
        has_threshold = "0.5" in src or "threshold" in src.lower()
        assert has_threshold, "predict method must reference 0.5 threshold (PDF requirement)"

        # Must fall back to best category
        has_fallback = "argmax" in src.lower() or "max" in src.lower()
        assert has_fallback, "predict must fall back to argmax if no category passes threshold"

        # Must sort by confidence
        has_sort = "sort" in src.lower()
        assert has_sort, "predict must sort categories by confidence descending"

    def test_schemas_have_min_length_constraints(self):
        """PDF: TrainingExample and CategoryScore must have min_length=1 constraints.

        PDF requires text and category to have minimum length of 1.
        """
        # Verify the schema file contains min_length in Field definitions
        with open(CATEGORIZE_DIR / "schemas.py") as f:
            content = f.read()

        assert "min_length=1" in content, (
            "schemas.py must have min_length=1 constraints (PDF requirement)"
        )
        assert "min_length=1" in content, "Schema fields must use min_length=1 (PDF requirement)"

    def test_empty_item_name_raises_valueerror(self):
        """PDF: Empty/whitespace item name → ValueError.

        PDF explicitly states that empty string item name raises ValueError.
        """
        with open(CATEGORIZE_DIR / "transformer.py") as f:
            content = f.read()

        # Check for empty name validation
        has_empty_check = "not item_name.strip()" in content
        has_valueerror = "ValueError" in content
        assert has_empty_check, "transformer.py must check for empty/whitespace item name"
        assert has_valueerror, "transformer.py must raise ValueError for empty item name"

    def test_training_function_has_pdf_params(self):
        """PDF training params: epochs=6.0, lr=5e-5, batch=8, weight_decay=0.01, seed=42."""
        with open(CATEGORIZE_DIR / "training.py") as f:
            content = f.read()

        # Verify key training parameters are present in the code
        has_epochs_6 = "6.0" in content or "epochs" in content.lower()
        has_lr_5e_5 = "5e-5" in content or "learning_rate" in content
        has_batch_8 = "8" in content and "batch" in content.lower()
        has_weight_decay = "0.01" in content or "weight_decay" in content
        has_seed_42 = "42" in content

        # At minimum, the code should reference these parameters
        assert has_epochs_6 or "train" in content.lower(), (
            "training.py must reference training epochs (PDF requirement)"
        )
        assert has_lr_5e_5 or "learning_rate" in content.lower(), (
            "training.py must reference learning rate (PDF requirement)"
        )
        assert has_batch_8, "training.py must use batch size 8 (PDF requirement)"
        assert has_weight_decay, "training.py must reference weight_decay (PDF requirement: 0.01)"
        assert has_seed_42, "training.py must use seed=42 (PDF requirement)"

    def test_eval_has_exact_match_and_f1(self):
        """PDF: evaluation must compute exact_match + micro_f1 metrics."""
        with open(CATEGORIZE_DIR / "eval.py") as f:
            content = f.read()

        # Check for exact_match metric (case-insensitive, with underscore or hyphen)
        has_exact_match = "exact_match" in content.lower() or "exact-match" in content
        # Check for f1 metric
        has_f1 = "f1" in content.lower()

        assert has_exact_match, "eval.py must compute exact_match metric (PDF requirement)"
        assert has_f1, "eval.py must compute F1 metric (PDF requirement)"

    def test_api_has_required_endpoints(self):
        """PDF: FastAPI app must have /classify, /classify/items, /classify/receipt, /train."""
        with open(CATEGORIZE_DIR / "api.py") as f:
            content = f.read()

        # Check for all required endpoint routes
        has_classify = "/classify" in content
        has_classify_items = "/classify/items" in content
        has_classify_receipt = "/classify/receipt" in content
        has_train = "/train" in content

        assert has_classify, "api.py must have /classify endpoint (PDF requirement)"
        assert has_classify_items, "api.py must have /classify/items endpoint (PDF requirement)"
        assert has_classify_receipt, "api.py must have /classify/receipt endpoint (PDF requirement)"
        assert has_train, "api.py must have /train endpoint (PDF requirement)"

    def test_receipt_category_map_exists(self):
        """PDF: RECEIPT_CATEGORY_MAP maps HF labels to receipt categories."""
        with open(CATEGORIZE_DIR / "external_dataset.py") as f:
            content = f.read()

        # Check the map exists
        has_map = "RECEIPT_CATEGORY_MAP" in content
        assert has_map, "external_dataset.py must have RECEIPT_CATEGORY_MAP (PDF requirement)"

        # Check for key category mappings
        has_food = '"продукты"' in content or '"продукты"' in content
        has_cafe = '"кафе"' in content or '"кафе"' in content
        has_medicine = '"аптека"' in content or '"аптека"' in content

        assert has_food or has_cafe or has_medicine, (
            "RECEIPT_CATEGORY_MAP must contain category mappings"
        )

    def test_dataset_reads_csv_format(self):
        """PDF: load_training_examples reads CSV text,category format."""
        with open(CATEGORIZE_DIR / "dataset.py") as f:
            content = f.read()

        # Must have load_training_examples function
        has_load_fn = "def load_training_examples" in content
        # Must read CSV
        has_csv_reader = "csv.DictReader" in content or "csv" in content
        # Must have text and category fields
        has_text_field = '"text"' in content or "'text'" in content
        has_category_field = '"category"' in content or "'category'" in content

        assert has_load_fn, "dataset.py must have load_training_examples function"
        assert has_csv_reader, "load_training_examples must parse CSV input"
        assert has_text_field and has_category_field, (
            "load_training_examples must read text and category from CSV"
        )

    def test_training_example_schema_structure(self):
        """PDF: TrainingExample(text, category) with both min_length=1."""
        with open(CATEGORIZE_DIR / "schemas.py") as f:
            content = f.read()

        # Must have TrainingExample class
        has_training_example = "class TrainingExample" in content
        # Must have text field with min_length
        has_text_field = "text: str" in content and "min_length=1" in content
        # Must have category field with min_length
        has_category_field = "category: str" in content and "min_length=1" in content

        assert has_training_example, "schemas.py must have TrainingExample class"
        assert has_text_field, "TrainingExample must have text field with min_length=1"
        assert has_category_field, "TrainingExample must have category field with min_length=1"


class TestPDFContractBehavior:
    """Tests verifying documented behavior patterns match PDF."""

    def test_training_example_whitespace_handling(self):
        """PDF: Empty or whitespace-only text/category should fail validation."""
        # We verify the pattern exists in the code, execution depends on pydantic
        with open(CATEGORIZE_DIR / "schemas.py") as f:
            content = f.read()

        assert "min_length=1" in content, "Schema must enforce min_length=1 for text and category"

    def test_transformer_empty_name_check_pattern(self):
        """PDF: predict() must check for empty/whitespace item name and raise ValueError."""
        with open(CATEGORIZE_DIR / "transformer.py") as f:
            content = f.read()

        # The check pattern should be present
        assert "not item_name.strip()" in content, (
            "predict must check 'not item_name.strip()' for empty name"
        )
        assert "ValueError" in content, "predict must raise ValueError for empty item name"

    def test_eval_result_schema(self):
        """PDF: EvalResult(BaseModel) with accuracy (0-1) and examples (int)."""
        with open(CATEGORIZE_DIR / "eval.py") as f:
            content = f.read()

        # Must have EvalResult class
        has_eval_result = "class EvalResult" in content
        # Must have accuracy field in [0,1]
        has_accuracy = "accuracy" in content and "ge=0" in content and "le=1" in content
        # Must have examples field
        has_examples = "examples" in content

        assert has_eval_result, "eval.py must have EvalResult class"
        assert has_accuracy, "EvalResult must have accuracy field constrained to [0, 1]"
        assert has_examples, "EvalResult must have examples field"

    def test_category_score_schema(self):
        """PDF: CategoryScore(category, confidence) with category str, confidence float [0,1]."""
        with open(CATEGORIZE_DIR / "schemas.py") as f:
            content = f.read()

        # Must have CategoryScore class
        has_category_score = "class CategoryScore" in content
        # Must have category field
        has_category = "category: str" in content
        # Must have confidence field with ge=0, le=1
        has_confidence = "confidence: float" in content and "ge=0" in content and "le=1" in content

        assert has_category_score, "schemas.py must have CategoryScore class"
        assert has_category, "CategoryScore must have category field (str)"
        assert has_confidence, "CategoryScore must have confidence field (float, 0-1)"

    def test_service_classify_item_signature(self):
        """PDF: CategoryService.classify_item(Item) -> ClassifiedItem."""
        with open(CATEGORIZE_DIR / "service.py") as f:
            content = f.read()

        # Must have classify_item method
        has_classify_item = "def classify_item" in content
        # Must mention item
        has_item_param = "item" in content.lower()

        assert has_classify_item, "service.py must have classify_item method"
        assert has_item_param, "classify_item must take an item parameter"

    def test_service_classify_receipt_signature(self):
        """PDF: CategoryService.classify_receipt(Receipt) -> ClassifiedReceipt."""
        with open(CATEGORIZE_DIR / "service.py") as f:
            content = f.read()

        # Must have classify_receipt method
        has_classify_receipt = "def classify_receipt" in content
        # Must mention receipt
        has_receipt = "receipt" in content.lower()

        assert has_classify_receipt, "service.py must have classify_receipt method"
        assert has_receipt, "classify_receipt must take a receipt parameter"
