import streamlit as st
import pandas as pd
import plotly.express as px
import duckdb
from google import genai

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Data Analyst Agent")

st.write(
    "Upload your CSV or Excel dataset and analyze your data."
)

uploaded_file = st.file_uploader(
    "📂 Choose a CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    try:

        if uploaded_file.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        st.success("✅ Dataset uploaded successfully!")
        duckdb.register("sales", df)

    except Exception as e:

        st.error(f"❌ Error reading file: {e}")
        st.stop()

    st.subheader("📊 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("📈 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.subheader("📋 Column Information")

    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        column_info,
        use_container_width=True
    )

    st.subheader("📋 Statistical Summary")

    numeric_df = df.select_dtypes(
        include="number"
    )

    if not numeric_df.empty:

        st.dataframe(
            numeric_df.describe(),
            use_container_width=True
        )

    else:

        st.info(
            "No numeric columns found for statistical analysis."
        )

    st.subheader("📊 Data Visualization")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numeric_columns) > 0:

        selected_column = st.selectbox(
            "Select a numeric column:",
            numeric_columns
        )

        fig = px.histogram(
            df,
            x=selected_column,
            title=f"Distribution of {selected_column}",
            color_discrete_sequence=["#636EFA"]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "⚠️ Your dataset does not contain numeric columns."
        )

    if "Product" in df.columns and "Quantity" in df.columns:

        st.subheader("📦 Quantity by Product")

        product_data = (
            df.groupby("Product", as_index=False)["Quantity"]
            .sum()
            .sort_values("Quantity", ascending=False)
        )

        fig1 = px.bar(
            product_data,
            x="Product",
            y="Quantity",
            title="Quantity by Product",
            color="Quantity",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

    if "Category" in df.columns and "Revenue" in df.columns:

        st.subheader("💰 Revenue by Category")

        category_data = (
            df.groupby("Category", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=False)
        )

        fig2 = px.bar(
            category_data,
            x="Category",
            y="Revenue",
            title="Revenue by Category",
            color="Revenue",
            color_continuous_scale="Greens"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    st.subheader("🗄️ SQL Data Analysis")

    sql_query = st.text_area(
        "Enter SQL query:",
        value="SELECT * FROM sales LIMIT 5"
    )

    if sql_query:

        try:

            result = duckdb.query(sql_query).df()

            st.dataframe(
                result,
                use_container_width=True
            )

        except Exception as e:

            st.error(f"SQL Error: {e}")

    st.subheader("🤖 Ask Your Data")

    question = st.text_input(
        "Ask a question about your dataset:",
        placeholder="Example: Which product has the highest quantity?"
    )

    if question:

        question_lower = question.lower().strip()

        if "highest" in question_lower and "quantity" in question_lower:

            if "Product" in df.columns and "Quantity" in df.columns:

                result = df.loc[df["Quantity"].idxmax()]

                st.success(
                    f"The product with the highest quantity is "
                    f"{result['Product']} with "
                    f"{result['Quantity']} units."
                )

            else:

                st.error(
                    "Your dataset must contain Product and Quantity columns."
                )

        elif "lowest" in question_lower and "quantity" in question_lower:

            if "Product" in df.columns and "Quantity" in df.columns:

                result = df.loc[df["Quantity"].idxmin()]

                st.success(
                    f"The product with the lowest quantity is "
                    f"{result['Product']} with "
                    f"{result['Quantity']} units."
                )

            else:

                st.error(
                    "Your dataset must contain Product and Quantity columns."
                )

        elif "total" in question_lower and "quantity" in question_lower:

            if "Quantity" in df.columns:

                total_quantity = df["Quantity"].sum()

                st.success(
                    f"The total quantity is {total_quantity} units."
                )

            else:

                st.error(
                    "Your dataset must contain Quantity column."
                )

        else:

            st.warning(
                "I couldn't understand that question yet."
            )

            st.write("Try:")
            st.write("- Which product has the highest quantity?")
            st.write("- Which product has the lowest quantity?")
            st.write("- What is the total quantity?")
        st.subheader("🧠 RAG - AI Data Assistant")

    rag_question = st.text_input(
        "Ask the AI about your dataset:",
        placeholder="Example: Give me a summary of this dataset.",
        key="rag_question"
    )

    if rag_question:

        data_context = df.head(100).to_string(index=False)

        prompt = f"""
You are an AI Data Analyst.

Analyze the following dataset and answer the user's question.

Dataset:
{data_context}

Dataset shape:
Rows: {df.shape[0]}
Columns: {df.shape[1]}

Column names:
{list(df.columns)}

User question:
{rag_question}

Give a clear, simple and useful answer.

For a summary, include:
- Number of rows and columns
- Important columns
- Missing values
- Main numerical information
- Important observations

Do not invent information.
"""

        try:
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt
            )

            if response and response.text:
                st.success("🤖 AI Answer")
                st.write(response.text)
            else:
                st.warning("AI returned an empty response.")

        except Exception as e:
            st.error(f"AI Error: {e}")