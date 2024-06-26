# from enum import StrEnum
from typing import Union

import pandas as pd
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    OrdinalEncoder,
    FunctionTransformer,
    TargetEncoder,
    LabelEncoder,
)


def preprocessing_for_logreg(pca_components: Union[bool, int] = False):
    no_transformer = "passthrough"

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]
    )

    numerical_transformer = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    transformers = [
        ("no_transform", no_transformer, make_column_selector(dtype_include=["bool"])),
        ("num", numerical_transformer, make_column_selector(dtype_include="number")),
        (
            "cat",
            categorical_transformer,
            make_column_selector(dtype_include=["object", "category"]),
        ),
    ]

    encoded_transformer = ColumnTransformer(
        transformers=transformers, remainder="passthrough"
    )

    if pca_components:
        pca = PCA(n_components=pca_components if isinstance(pca_components, int) else 3)
        return [("preprocessing", encoded_transformer), ("pca", pca)]
    else:
        return encoded_transformer


def convert_to_category(df):
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].astype("category")
    return df


def convert_to_category_label_encoder(df):
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].astype("category")

    # Identify categorical columns
    categorical_columns = [
        col
        for col in df.columns
        if df[col].dtype == "object" or df[col].dtype == "category"
    ]

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    # Encode each categorical column
    for col in categorical_columns:
        # Combine training and test column values
        # combined = pd.concat([features[col], test_features[col]], axis=0)

        # Fit LabelEncoder on combined data
        label_encoder.fit(df[col])

        # Transform both training and testing data
        df[col] = label_encoder.transform(df[col])
        # test_features[col] = label_encoder.transform(test_features[col])

    return df


def preprocessing_for_xgboost(
    use_categorical_feature=False,
    use_target_encoding=False,
    target=None,
    use_numerical_cats=False,
):
    if use_numerical_cats:
        categorical_transformer = Pipeline([("ordinal", OrdinalEncoder())])
    elif use_categorical_feature:
        return FunctionTransformer(convert_to_category, validate=False)

    elif use_target_encoding:
        if target is None:
            raise ValueError("Target variable must be provided for target encoding")

        categorical_transformer = Pipeline(
            [
                (
                    "target_encoder",
                    TargetEncoder(
                        cols=make_column_selector(dtype_include=["object", "category"]),
                        y=target,
                    ),
                )
            ]
        )

    else:
        categorical_transformer = Pipeline(
            [("onehot", OneHotEncoder(handle_unknown="error", drop="if_binary"))]
        )

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                categorical_transformer,
                make_column_selector(dtype_include=["object", "category"]),
            ),
        ]
    )

    return preprocessor


type_transform_categorical_options = Union["ADSasd"]


# class TransformCategoricalOptions(StrEnum):
#     use_categorical_feature = "use_categorical_feature"
#     # one_hot = "one_hot"
#     target_encoding = "target_encoding"
#


def preprocessing_for_xgboost_2():
    # Always set 'use_categorical_feature' for XGBoost to true
    # When we want to use alternatives like one hot just transform all categorical vars to bool etc.

    def convert_to_category(df):
        for col in df.select_dtypes(include=["object"]):
            df[col] = df[col].astype("category")
        return df

    raise NotImplementedError


def preprocessing_simplified():
    no_transformer = "passthrough"

    categorical_transformer = Pipeline([("ordinal", OrdinalEncoder())])
    preprocessor = ColumnTransformer(
        [
            (
                "no_transform",
                no_transformer,
                make_column_selector(dtype_include=["number", "bool"]),
            ),
            (
                "cat",
                categorical_transformer,
                make_column_selector(dtype_include=["object", "category"]),
            ),
        ]
    )

    return preprocessor


def preprocessing_for_svm(enable_pca=False, pca_components=None):
    no_transformer = "passthrough"

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]
    )

    numerical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),  # Scaling is important for SVM
        ]
    )

    transformers = [
        ("no_transform", no_transformer, make_column_selector(dtype_include=["bool"])),
        ("num", numerical_transformer, make_column_selector(dtype_include="number")),
        (
            "cat",
            categorical_transformer,
            make_column_selector(dtype_include=["object", "category"]),
        ),
    ]

    preprocessor = ColumnTransformer(transformers)

    return preprocessor


def preprocessing_for_lgbm_OLD():
    # def categorical_label_encoder(df):
    #     label_encoder = LabelEncoder()
    #     categorical_columns = [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype == 'category']
    #
    #     for col in categorical_columns:
    #         # Fit and transform the data to itself, essentially training and transforming on the same data
    #         df[col] = label_encoder.fit_transform(df[col])
    #
    #     return df

    categorical_transformer = Pipeline(
        [("onehot", OrdinalEncoder().set_output(transform="pandas"))]
    )

    # categorical_transformer = Pipeline([
    #     ('encoder', categorical_label_encoder)
    # ])

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                categorical_transformer,
                make_column_selector(dtype_include=["object", "category"]),
            ),
        ]
    )

    return preprocessor


def preprocessing_for_lgbm():
    categorical_transformer = Pipeline([("ordinal", OrdinalEncoder())])

    return FunctionTransformer(convert_to_category, validate=False)

    # return preprocessor
