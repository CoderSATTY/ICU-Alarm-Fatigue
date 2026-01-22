import gradio as gr
from agentic_mapping import generate_final_output, get_low_urgency, get_medium_urgency, get_high_urgency

custom_css = """
body, .gradio-container { font-size: 20px !important; }
.gr-markdown { font-size: 22px !important; line-height: 1.6 !important; }
#summary_output .prose, #summary_output .prose * { font-size: 24px !important; line-height: 1.8 !important; }
button { font-size: 20px !important; padding: 8px 14px !important; }
input, textarea { font-size: 20px !important; }
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🏥 ICU Alarm Monitor")
    
    with gr.Row():
        alarm_name_input = gr.Textbox(label="Alarm Name", placeholder="e.g., APNEA, AC POWER LOSS")
        urgency_input = gr.Textbox(label="Urgency (optional)", placeholder="Low, Medium, High")
    
    search_button = gr.Button("🔍 Analyze Alarm", variant="primary")
    
    gr.Markdown("### Filter by Urgency")
    with gr.Row():
        low_btn = gr.Button("Low Urgency")
        med_btn = gr.Button("Medium Urgency")
        high_btn = gr.Button("High Urgency")
    
    summary_output = gr.Markdown(elem_id="summary_output")
    
    search_button.click(fn=generate_final_output, inputs=[alarm_name_input, urgency_input], outputs=summary_output, show_progress="full")
    low_btn.click(fn=get_low_urgency, outputs=summary_output, show_progress="full")
    med_btn.click(fn=get_medium_urgency, outputs=summary_output, show_progress="full")
    high_btn.click(fn=get_high_urgency, outputs=summary_output, show_progress="full")

demo.launch(share=True)