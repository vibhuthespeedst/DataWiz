import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import io
import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="Data Insights AI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Initialize LLM with API key from environment
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.3,
            api_key=api_key
        )
    else:
        llm = None
        st.error("❌ GOOGLE_API_KEY not found in environment variables. Please add it to your .env file")
except Exception as e:
    st.error(f"❌ Error configuring API: {str(e)}")
    llm = None

# Sidebar for file upload
with st.sidebar:
    st.title("🔧 Configuration")
    
    # Show API status
    if llm is not None:
        st.success("✅ Google API Key loaded successfully!")
    else:
        st.error("❌ API Key not configured")
        st.info("💡 Add GOOGLE_API_KEY to your .env file")
    
    st.divider()
    
    # File upload
    st.title("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset to get instant insights"
    )
    
    if uploaded_file is not None:
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                # Check if openpyxl is available for Excel files
                try:
                    df = pd.read_excel(uploaded_file)
                except ImportError:
                    st.error("❌ Missing dependency for Excel files. Please install: `pip install openpyxl`")
                    st.stop()
            
            st.session_state.data = df
            st.success(f"✅ File uploaded successfully!")
            st.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            
        except Exception as e:
            if "openpyxl" in str(e):
                st.error("❌ Missing dependency for Excel files. Please install: `pip install openpyxl`")
                st.info("💡 Or upload a CSV file instead")
            else:
                st.error(f"❌ Error reading file: {str(e)}")

# Main app
st.title("📊 Data Insights AI Assistant")
st.markdown("*Get instant insights, interactive visualizations, and AI-powered analysis for your datasets*")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "💬 Chat Assistant", "🔍 Insights", "📊 Interactive Graphs"])

def generate_basic_insights(df):
    """Generate basic statistical insights about the dataset"""
    insights = []
    
    # Basic info
    insights.append(f"**Dataset Overview:**")
    insights.append(f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    insights.append(f"- Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    insights.append(f"\n**Column Types:**")
    insights.append(f"- Numeric: {len(numeric_cols)} columns")
    insights.append(f"- Categorical: {len(categorical_cols)} columns")
    insights.append(f"- DateTime: {len(datetime_cols)} columns")
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    cols_with_missing = missing_data[missing_data > 0]
    
    if len(cols_with_missing) > 0:
        insights.append(f"\n**Missing Values:**")
        for col in cols_with_missing.index[:5]:  # Show top 5
            insights.append(f"- {col}: {cols_with_missing[col]} ({missing_percent[col]:.1f}%)")
    else:
        insights.append(f"\n**Missing Values:** None ✅")
    
    # Duplicates
    duplicates = df.duplicated().sum()
    insights.append(f"\n**Duplicate Rows:** {duplicates} ({duplicates/len(df)*100:.1f}%)")
    
    # Numeric column insights
    if numeric_cols:
        insights.append(f"\n**Numeric Columns Summary:**")
        for col in numeric_cols[:3]:  # Top 3 numeric columns
            col_data = df[col].dropna()
            insights.append(f"- {col}: Mean={col_data.mean():.2f}, Std={col_data.std():.2f}, Range=[{col_data.min():.2f}, {col_data.max():.2f}]")
    
    return "\n".join(insights)

def create_overview_charts(df):
    """Create overview charts for the dataset"""
    charts = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 1. Missing values heatmap
    if df.isnull().sum().sum() > 0:
        fig_missing = px.imshow(
            df.isnull().astype(int),
            title="Missing Values Pattern",
            color_continuous_scale='Reds',
            aspect='auto'
        )
        fig_missing.update_layout(height=400)
        charts.append(("Missing Values Heatmap", fig_missing))
    
    # 2. Correlation heatmap for numeric columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlation Matrix",
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(height=500)
        charts.append(("Correlation Matrix", fig_corr))
    
    # 3. Distribution of numeric columns
    if len(numeric_cols) > 0:
        fig_dist = make_subplots(
            rows=(len(numeric_cols[:4]) + 1) // 2, 
            cols=2,
            subplot_titles=numeric_cols[:4]
        )
        
        for i, col in enumerate(numeric_cols[:4]):
            row = i // 2 + 1
            col_num = i % 2 + 1
            fig_dist.add_trace(
                go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
                row=row, col=col_num
            )
        
        fig_dist.update_layout(height=400, title_text="Distribution of Numeric Columns")
        charts.append(("Numeric Distributions", fig_dist))
    
    return charts

# Tab 1: Overview
with tab1:
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Display basic info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing %", f"{(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%")
        with col4:
            st.metric("Duplicates", f"{df.duplicated().sum():,}")
        
        # Show data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show data types
        st.subheader("🏷️ Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null %': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(col_info, use_container_width=True)
        
        # Quick overview charts
        st.subheader("📊 Quick Overview")
        charts = create_overview_charts(df)
        
        for chart_name, fig in charts:
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.info("👆 Please upload a file in the sidebar to see the overview")

# Tab 2: Chat Assistant
with tab2:
    if st.session_state.data is not None and llm is not None:
        df = st.session_state.data
        
        st.subheader("💬 Chat with your Data")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask anything about your data..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            # Prepare context about the data
            data_context = f"""
            You are analyzing a dataset with the following characteristics:
            - Shape: {df.shape[0]} rows, {df.shape[1]} columns
            - Columns: {list(df.columns)}
            - Data types: {dict(df.dtypes)}
            - Missing values: {dict(df.isnull().sum())}
            - Basic statistics: {df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else 'No numeric columns'}
            
            Sample data (first 3 rows):
            {df.head(3).to_string()}
            
            User question: {prompt}
            
            Please provide insights, suggestions, or analysis based on this data. If the user asks about specific columns or wants recommendations for ML models, provide detailed and actionable advice.
            """
            
            try:
                # Get response from LLM
                response = llm.invoke([HumanMessage(content=data_context)])
                
                # Add to memory
                st.session_state.memory.chat_memory.add_user_message(prompt)
                st.session_state.memory.chat_memory.add_ai_message(response.content)
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response.content})
                st.chat_message("assistant").write(response.content)
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                st.chat_message("assistant").write(error_msg)
    
    elif st.session_state.data is None:
        st.info("👆 Please upload a file in the sidebar to start chatting")
    elif llm is None:
        st.info("🔑 Please enter your Google API Key in the sidebar to enable chat")

# Tab 3: Insights
with tab3:
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.subheader("🔍 Detailed Data Insights")
        
        # Generate and display insights
        insights = generate_basic_insights(df)
        st.markdown(insights)
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.info("No numeric columns found for statistical summary")
        
        # Categorical analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            st.subheader("🏷️ Categorical Analysis")
            
            for col in categorical_cols[:5]:  # Show top 5 categorical columns
                st.write(f"**{col}**")
                value_counts = df[col].value_counts().head(10)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.dataframe(value_counts)
                with col2:
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Top Values in {col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # ML Recommendations
        if llm is not None:
            st.subheader("🤖 ML Model Recommendations")
            
            ml_context = f"""
            Based on this dataset analysis:
            - Shape: {df.shape[0]} rows, {df.shape[1]} columns
            - Numeric columns: {len(numeric_cols)}
            - Categorical columns: {len(categorical_cols)}
            - Missing values: {df.isnull().sum().sum()} total
            
            Please recommend suitable preprocessing steps for this dataset. Consider:
            1. Recommended preprocessing steps for this dataset

            
            Keep recommendations practical and actionable.
            """
            
            try:
                response = llm.invoke([HumanMessage(content=ml_context)])
                st.markdown(response.content)
            except Exception as e:
                st.error(f"Error generating ML recommendations: {str(e)}")
        
    else:
        st.info("👆 Please upload a file in the sidebar to see insights")

# Tab 4: Interactive Graphs
with tab4:
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.subheader("📊 Interactive Visualizations")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        all_cols = list(df.columns)
        
        # Graph configuration
        graph_type = st.selectbox(
            "Select Graph Type",
            ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Violin Plot", "Heatmap", "Pair Plot"]
        )
        
        if graph_type == "Scatter Plot":
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", numeric_cols)
                color_by = st.selectbox("Color by", [None] + categorical_cols + numeric_cols)
            with col2:
                y_axis = st.selectbox("Y-axis", numeric_cols)
                size_by = st.selectbox("Size by", [None] + numeric_cols)
            
            if x_axis and y_axis:
                fig = px.scatter(
                    df, x=x_axis, y=y_axis,
                    color=color_by if color_by else None,
                    size=size_by if size_by else None,
                    title=f"{y_axis} vs {x_axis}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif graph_type == "Line Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", all_cols)
                color_by = st.selectbox("Color by", [None] + categorical_cols)
            with col2:
                y_axis = st.selectbox("Y-axis", numeric_cols)
            
            if x_axis and y_axis:
                fig = px.line(
                    df, x=x_axis, y=y_axis,
                    color=color_by if color_by else None,
                    title=f"{y_axis} over {x_axis}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif graph_type == "Bar Chart":
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", categorical_cols + numeric_cols)
                color_by = st.selectbox("Color by", [None] + categorical_cols)
            with col2:
                y_axis = st.selectbox("Y-axis", numeric_cols)
            
            if x_axis and y_axis:
                fig = px.bar(
                    df, x=x_axis, y=y_axis,
                    color=color_by if color_by else None,
                    title=f"{y_axis} by {x_axis}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif graph_type == "Histogram":
            col1, col2 = st.columns(2)
            with col1:
                column = st.selectbox("Column", numeric_cols)
                color_by = st.selectbox("Color by", [None] + categorical_cols)
            with col2:
                bins = st.slider("Number of bins", 10, 100, 30)
            
            if column:
                fig = px.histogram(
                    df, x=column,
                    color=color_by if color_by else None,
                    nbins=bins,
                    title=f"Distribution of {column}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif graph_type == "Box Plot":
            col1, col2 = st.columns(2)
            with col1:
                y_axis = st.selectbox("Y-axis (numeric)", numeric_cols)
            with col2:
                x_axis = st.selectbox("X-axis (categorical)", [None] + categorical_cols)
            
            if y_axis:
                fig = px.box(
                    df, y=y_axis, x=x_axis if x_axis else None,
                    title=f"Box Plot of {y_axis}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif graph_type == "Violin Plot":
            col1, col2 = st.columns(2)
            with col1:
                y_axis = st.selectbox("Y-axis (numeric)", numeric_cols)
            with col2:
                x_axis = st.selectbox("X-axis (categorical)", [None] + categorical_cols)
            
            if y_axis:
                fig = px.violin(
                    df, y=y_axis, x=x_axis if x_axis else None,
                    title=f"Violin Plot of {y_axis}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif graph_type == "Heatmap":
            if len(numeric_cols) > 1:
                selected_cols = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:10])
                if selected_cols:
                    corr_matrix = df[selected_cols].corr()
                    fig = px.imshow(
                        corr_matrix,
                        title="Correlation Heatmap",
                        color_continuous_scale='RdBu',
                        zmin=-1, zmax=1
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for heatmap")
        
        elif graph_type == "Pair Plot":
            if len(numeric_cols) > 1:
                selected_cols = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:4])
                color_by = st.selectbox("Color by", [None] + categorical_cols)
                
                if selected_cols and len(selected_cols) > 1:
                    # Create subplot pair plot
                    n_cols = len(selected_cols)
                    fig = make_subplots(
                        rows=n_cols, cols=n_cols,
                        subplot_titles=[f"{col1} vs {col2}" for col1 in selected_cols for col2 in selected_cols]
                    )
                    
                    for i, col1 in enumerate(selected_cols):
                        for j, col2 in enumerate(selected_cols):
                            if i == j:
                                # Diagonal: histogram
                                fig.add_trace(
                                    go.Histogram(x=df[col1], showlegend=False),
                                    row=i+1, col=j+1
                                )
                            else:
                                # Off-diagonal: scatter
                                fig.add_trace(
                                    go.Scatter(
                                        x=df[col2], y=df[col1],
                                        mode='markers',
                                        showlegend=False
                                    ),
                                    row=i+1, col=j+1
                                )
                    
                    fig.update_layout(height=600, title_text="Pair Plot")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need at least 2 numeric columns for pair plot")
        
        # Data filtering section
        st.subheader("🎛️ Data Filtering")
        
        # Filters
        filters = {}
        col1, col2 = st.columns(2)
        
        with col1:
            # Numeric filters
            st.write("**Numeric Filters**")
            for col in numeric_cols[:3]:
                min_val, max_val = float(df[col].min()), float(df[col].max())
                # Handle case where min and max are the same
                if min_val == max_val:
                    st.write(f"**{col}**: {min_val} (constant value)")
                    filters[col] = (min_val, max_val)
                else:
                    filters[col] = st.slider(f"{col}", min_val, max_val, (min_val, max_val))
        
        with col2:
            # Categorical filters
            st.write("**Categorical Filters**")
            for col in categorical_cols[:3]:
                unique_vals = df[col].unique()
                filters[col] = st.multiselect(f"{col}", unique_vals, default=unique_vals)
        
        # Apply filters
        filtered_df = df.copy()
        for col, filter_val in filters.items():
            if col in numeric_cols:
                # Skip filtering if min and max are the same (constant column)
                if filter_val[0] != filter_val[1]:
                    filtered_df = filtered_df[
                        (filtered_df[col] >= filter_val[0]) & 
                        (filtered_df[col] <= filter_val[1])
                    ]
            elif col in categorical_cols and filter_val:
                filtered_df = filtered_df[filtered_df[col].isin(filter_val)]
        
        st.write(f"**Filtered Data Shape:** {filtered_df.shape[0]} rows × {filtered_df.shape[1]} columns")
        st.dataframe(filtered_df.head(), use_container_width=True)
        
    else:
        st.info("👆 Please upload a file in the sidebar to create interactive graphs")

# Footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit, LangChain, and Google's Gemini AI*")