"""
Main Gradio application for the Hugging Face Model Ecosystem Navigator.
"""
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import os

from data_loader import ModelDataLoader
from embeddings import ModelEmbedder
from dimensionality_reduction import DimensionReducer


class ModelNavigatorApp:
    """Main application class for the model navigator."""
    
    def __init__(self):
        self.data_loader = ModelDataLoader()
        self.embedder: Optional[ModelEmbedder] = None
        self.reducer: Optional[DimensionReducer] = None
        self.df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.reduced_embeddings: Optional[np.ndarray] = None
        self.current_filtered_df: Optional[pd.DataFrame] = None
        
    def load_initial_data(self, sample_size: int = 10000):
        """Load initial sample of data."""
        print("Loading initial data...")
        self.df = self.data_loader.load_data(sample_size=sample_size)
        self.df = self.data_loader.preprocess_for_embedding(self.df)
        return f"Loaded {len(self.df)} models"
    
    def generate_visualization(
        self,
        color_by: str = "library_name",
        size_by: str = "downloads",
        min_downloads: int = 0,
        min_likes: int = 0,
        search_query: str = "",
        selected_libraries: list = None,
        selected_pipeline_tags: list = None,
        use_cache: bool = True
    ) -> Tuple[go.Figure, pd.DataFrame]:
        """
        Generate interactive visualization.
        
        Returns:
            Plotly figure and filtered dataframe
        """
        if self.df is None or len(self.df) == 0:
            return go.Figure(), pd.DataFrame()
        
        # Filter data
        filtered_df = self.data_loader.filter_data(
            df=self.df,
            min_downloads=min_downloads,
            min_likes=min_likes,
            libraries=selected_libraries if selected_libraries else None,
            pipeline_tags=selected_pipeline_tags if selected_pipeline_tags else None,
            search_query=search_query if search_query else None
        )
        
        if len(filtered_df) == 0:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No models match the selected filters",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return empty_fig, filtered_df
        
        # Limit to reasonable size for performance
        max_points = 5000
        if len(filtered_df) > max_points:
            filtered_df = filtered_df.sample(n=max_points, random_state=42)
            print(f"Sampled {max_points} models for visualization")
        
        # Get indices for filtered data
        filtered_indices = filtered_df.index.tolist()
        
        # Generate or load embeddings
        cache_file = "embeddings_cache.pkl"
        if use_cache and os.path.exists(cache_file) and self.embeddings is None:
            try:
                if self.embedder is None:
                    self.embedder = ModelEmbedder()
                self.embeddings = self.embedder.load_embeddings(cache_file)
            except Exception as e:
                print(f"Could not load cached embeddings: {e}")
                pass
        
        if self.embeddings is None:
            if self.embedder is None:
                self.embedder = ModelEmbedder()
            
            # Generate embeddings for all data
            texts = self.df['combined_text'].tolist()
            self.embeddings = self.embedder.generate_embeddings(texts)
            
            if use_cache:
                self.embedder.save_embeddings(self.embeddings, cache_file)
        
        # Get embeddings for filtered data
        filtered_embeddings = self.embeddings[filtered_indices]
        
        # Reduce dimensions
        if self.reducer is None:
            self.reducer = DimensionReducer(method="umap", n_components=2)
        
        reduced_cache_file = "reduced_embeddings_cache.npy"
        if use_cache and os.path.exists(reduced_cache_file):
            try:
                self.reduced_embeddings = np.load(reduced_cache_file, allow_pickle=True)
                if len(self.reduced_embeddings) != len(self.df):
                    self.reduced_embeddings = None
            except Exception as e:
                print(f"Could not load cached reduced embeddings: {e}")
                pass
        
        if self.reduced_embeddings is None or len(self.reduced_embeddings) != len(self.df):
            self.reduced_embeddings = self.reducer.fit_transform(self.embeddings)
            if use_cache:
                np.save(reduced_cache_file, self.reduced_embeddings)
        
        filtered_reduced = self.reduced_embeddings[filtered_indices]
        
        # Prepare data for plotting
        plot_df = filtered_df.copy()
        plot_df['x'] = filtered_reduced[:, 0]
        plot_df['y'] = filtered_reduced[:, 1]
        
        # Color mapping
        if color_by in plot_df.columns:
            color_values = plot_df[color_by].fillna('Unknown')
        else:
            color_values = pd.Series(['All Models'] * len(plot_df))
        
        # Size mapping
        if size_by and size_by != "None" and size_by in plot_df.columns:
            size_values = plot_df[size_by].fillna(0)
            # Normalize sizes
            if size_values.max() > 0:
                size_values = 5 + 15 * (size_values / size_values.max())
            else:
                size_values = pd.Series([10] * len(plot_df))
        else:
            size_values = pd.Series([10] * len(plot_df))
        
        # Create hover text
        hover_texts = []
        for idx, row in plot_df.iterrows():
            hover = f"<b>{row.get('model_id', 'Unknown')}</b><br>"
            hover += f"Library: {row.get('library_name', 'N/A')}<br>"
            hover += f"Pipeline: {row.get('pipeline_tag', 'N/A')}<br>"
            hover += f"Downloads: {row.get('downloads', 0):,}<br>"
            hover += f"Likes: {row.get('likes', 0):,}"
            hover_texts.append(hover)
        
        # Create plotly figure
        fig = go.Figure()
        
        # Store model IDs with indices for click handling
        model_id_map = {i: row.get('model_id', 'Unknown') for i, row in plot_df.iterrows()}
        
        # Group by color if categorical
        is_categorical = len(color_values) > 0 and isinstance(color_values.iloc[0], str)
        
        if is_categorical and color_by in plot_df.columns:
            unique_colors = color_values.unique()
            colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel
            color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_colors)}
            
            for color_val in unique_colors:
                mask = color_values == color_val
                subset_df = plot_df[mask]
                subset_hover = [hover_texts[i] for i, m in enumerate(mask) if m]
                subset_sizes = size_values[mask]
                
                # Create customdata with model IDs for click handling
                subset_customdata = [
                    [int(idx), str(row.get('model_id', 'Unknown'))] 
                    for idx, row in subset_df.iterrows()
                ]
                
                fig.add_trace(go.Scatter(
                    x=subset_df['x'],
                    y=subset_df['y'],
                    mode='markers',
                    name=str(color_val)[:30],  # Truncate long names
                    marker=dict(
                        size=subset_sizes.values,
                        color=color_map[color_val],
                        opacity=0.7,
                        line=dict(width=0.5, color='white')
                    ),
                    text=subset_df['model_id'].tolist(),
                    customdata=subset_customdata,
                    hovertemplate='%{text}<br>Click for details<extra></extra>',
                    showlegend=True
                ))
        else:
            # Continuous color scale
            customdata = [
                [int(idx), str(row.get('model_id', 'Unknown'))] 
                for idx, row in plot_df.iterrows()
            ]
            
            fig.add_trace(go.Scatter(
                x=plot_df['x'],
                y=plot_df['y'],
                mode='markers',
                marker=dict(
                    size=size_values.values,
                    color=color_values.values,
                    colorscale='Viridis',
                    opacity=0.7,
                    line=dict(width=0.5, color='white'),
                    colorbar=dict(title=color_by)
                ),
                text=plot_df['model_id'].tolist(),
                customdata=customdata,
                hovertemplate='%{text}<br>Click for details<extra></extra>',
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Model Latent Space Navigator ({len(plot_df)} models)',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            hovermode='closest',
            template='plotly_white',
            height=700,
            clickmode='event+select'
        )
        
        return fig, filtered_df
    
    def get_model_details(self, model_id: str) -> str:
        """Get detailed information about a model."""
        if self.df is None:
            return "No data loaded"
        
        model = self.df[self.df.get('model_id', '') == model_id]
        if len(model) == 0:
            return f"Model '{model_id}' not found"
        
        model = model.iloc[0]
        
        details = f"# {model.get('model_id', 'Unknown')}\n\n"
        details += f"**Library:** {model.get('library_name', 'N/A')}\n\n"
        details += f"**Pipeline Tag:** {model.get('pipeline_tag', 'N/A')}\n\n"
        details += f"**Downloads:** {model.get('downloads', 0):,}\n\n"
        details += f"**Likes:** {model.get('likes', 0):,}\n\n"
        details += f"**Trending Score:** {model.get('trendingScore', 'N/A')}\n\n"
        
        if pd.notna(model.get('tags')):
            details += f"**Tags:** {model.get('tags', '')}\n\n"
        
        if pd.notna(model.get('licenses')):
            details += f"**License:** {model.get('licenses', '')}\n\n"
        
        if pd.notna(model.get('parent_model')):
            details += f"**Parent Model:** {model.get('parent_model', '')}\n\n"
        
        return details


def create_interface():
    """Create and launch Gradio interface."""
    app = ModelNavigatorApp()
    
    # Load initial data
    status = app.load_initial_data(sample_size=10000)
    print(status)
    
    with gr.Blocks(title="Anatomy of a Machine Learning Ecosystem", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Anatomy of a Machine Learning Ecosystem: 2 Million Models on Hugging Face
        
        **Authors:** Benjamin Laufer, Hamidah Oderinwale, Jon Kleinberg
        
        Many have observed that the development and deployment of generative machine learning (ML) and artificial intelligence (AI) models follow a distinctive pattern in which pre-trained models are adapted and fine-tuned for specific downstream tasks. However, there is limited empirical work that examines the structure of these interactions. This paper analyzes 1.86 million models on Hugging Face, a leading peer production platform for model development. Our study of model family trees -- networks that connect fine-tuned models to their base or parent -- reveals sprawling fine-tuning lineages that vary widely in size and structure. Using an evolutionary biology lens to study ML models, we use model metadata and model cards to measure the genetic similarity and mutation of traits over model families.
        
        **Read the full paper**: [arXiv:2508.06811](https://arxiv.org/abs/2508.06811)
        
        ---
        
        **How to use this navigator:**
        - Adjust filters to explore different subsets of models
        - Hover over points to see model information
        - Use color and size options to highlight different attributes
        - Similar models appear closer together in the latent space
        - Models are positioned based on their similarity (tags, pipeline, library, and model card content)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Filters")
                
                search_query = gr.Textbox(
                    label="Search Models",
                    placeholder="Search by model ID or tags...",
                    value=""
                )
                
                min_downloads = gr.Slider(
                    label="Min Downloads",
                    minimum=0,
                    maximum=1000000,
                    value=0,
                    step=1000
                )
                
                min_likes = gr.Slider(
                    label="Min Likes",
                    minimum=0,
                    maximum=10000,
                    value=0,
                    step=10
                )
                
                color_by = gr.Dropdown(
                    label="Color By",
                    choices=["library_name", "pipeline_tag", "downloads", "likes"],
                    value="library_name"
                )
                
                size_by = gr.Dropdown(
                    label="Size By",
                    choices=["downloads", "likes", "trendingScore", "None"],
                    value="downloads"
                )
                
                update_btn = gr.Button("Update Visualization", variant="primary")
            
            with gr.Column(scale=3):
                plot = gr.Plot(label="Model Latent Space")
                model_details = gr.Markdown(
                    value="**Instructions:** Use the filters above to explore models. Hover over points to see details, **click on a point** to view full model information and link to Hugging Face.",
                    label="Model Details"
                )
        
        def handle_plot_click(evt: gr.SelectData):
            """Handle plot click and show model details."""
            if evt is None or app.df is None:
                return "**Click on a model point to see details**"
            
            try:
                # Get the point index from the click event
                point_idx = evt.index
                if point_idx is None:
                    return "**Click on a model point to see details**"
                
                # Get the current filtered dataframe
                if app.current_filtered_df is not None and len(app.current_filtered_df) > 0:
                    filtered_df = app.current_filtered_df
                else:
                    # Fallback: use the full dataframe
                    filtered_df = app.df
                
                # Limit to max_points if needed
                if len(filtered_df) > 5000:
                    filtered_df = filtered_df.sample(n=5000, random_state=42)
                
                if point_idx < len(filtered_df):
                    model_row = filtered_df.iloc[point_idx]
                    model_id = model_row.get('model_id', 'Unknown')
                    
                    # Get full model details from the original dataframe
                    model = app.df[app.df.get('model_id', '') == model_id]
                    if len(model) == 0:
                        return f"**Model not found:** {model_id}"
                    
                    model = model.iloc[0]
                    hf_url = f"https://huggingface.co/{model_id}"
                    
                    details = f"""# {model_id}

**[View on Hugging Face]({hf_url})**

## Model Information

- **Library:** {model.get('library_name', 'N/A')}
- **Pipeline Tag:** {model.get('pipeline_tag', 'N/A')}
- **Downloads:** {model.get('downloads', 0):,}
- **Likes:** {model.get('likes', 0):,}
"""
                    if pd.notna(model.get('trendingScore')):
                        details += f"- **Trending Score:** {model.get('trendingScore', 0):.2f}\n\n"
                    else:
                        details += "\n"
                    
                    if pd.notna(model.get('tags')):
                        details += f"**Tags:** {model.get('tags', '')}\n\n"
                    if pd.notna(model.get('licenses')):
                        details += f"**License:** {model.get('licenses', '')}\n\n"
                    if pd.notna(model.get('parent_model')):
                        details += f"**Parent Model:** {model.get('parent_model', '')}\n\n"
                    
                    return details
                else:
                    return f"**Point index out of range:** {point_idx}"
            except Exception as e:
                import traceback
                return f"**Error loading model details:**\n```\n{str(e)}\n{traceback.format_exc()}\n```"
            
            return "**Click on a model point to see details**"
        
        def update_plot_and_store(color_by_val, size_by_val, min_dl, min_lk, search):
            fig, df = app.generate_visualization(
                color_by=color_by_val,
                size_by=size_by_val,
                min_downloads=int(min_dl),
                min_likes=int(min_lk),
                search_query=search
            )
            # Store the filtered dataframe for click handling
            app.current_filtered_df = df
            return fig
        
        update_btn.click(
            fn=update_plot_and_store,
            inputs=[color_by, size_by, min_downloads, min_likes, search_query],
            outputs=plot
        )
        
        # Handle plot clicks - Gradio's Plot component supports click events
        plot.select(
            fn=handle_plot_click,
            outputs=model_details
        )
        
        # Initial plot
        initial_fig, initial_df = app.generate_visualization()
        plot.value = initial_fig
        app.current_filtered_df = initial_df
    
    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
