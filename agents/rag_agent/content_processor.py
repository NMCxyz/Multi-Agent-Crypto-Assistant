import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

class ContentProcessor:
    """
    Processes the parsed content - summarizes images, creates llm based semantic chunks
    """
    def __init__(self, config):
        """
        Initialize the response generator.
        
        Args:
            llm: Large language model for image summarization
        """
        self.logger = logging.getLogger(__name__)
        self.summarizer_model = config.rag.summarizer_model     # temperature 0.5
        self.chunker_model = config.rag.chunker_model     # temperature 0.0
    
    def summarize_images(self, images: List[str]) -> List[str]:
        """
        Summarize images using the provided model.

        Args:
            images: List of image paths

        Returns:
            List of image summaries
        """

        prompt_template = """Describe the image in detail while keeping it concise and to the point.
                        For context, the image is part of either a crypto/financial research paper or a research paper
                        demonstrating the use of artificial intelligence techniques like
                        machine learning and deep learning in financial analysis or a financial report.
                        Be specific about graphs, such as bar plots if they are present in the image.
                        Only summarize what is present in the image, without adding any extra detail or comment.
                        Summarize the image only if it is related to the context, return 'non-informative' explicitly
                        if the image is of some button not relevant to the context."""

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_template},
                    {
                        "type": "image_url",
                        "image_url": {"url": "{image}"},
                    },
                ],
            )
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        summary_chain = prompt | self.summarizer_model | StrOutputParser()

        results = []
        for image in images:
            summary = summary_chain.invoke({"image": image})
            results.append(summary)

        return results
    
    def format_document_with_images(self, parsed_document: Any, image_summaries: List[str]) -> str:
        """
        Format the parsed document by replacing image placeholders with image summaries.
        
        Args:
            parsed_document: Parsed document from doc_parser
            image_summaries: List of image summaries
            
        Returns:
            Formatted document text with image summaries
        """
        IMAGE_PLACEHOLDER = "<!-- image_placeholder -->"
        PAGE_BREAK_PLACEHOLDER = "<!-- page_break -->"
        
        formatted_parsed_document = parsed_document.export_to_markdown(
            page_break_placeholder=PAGE_BREAK_PLACEHOLDER, 
            image_placeholder=IMAGE_PLACEHOLDER
        )
        
        formatted_document = self._replace_occurrences(
            formatted_parsed_document, 
            IMAGE_PLACEHOLDER, 
            image_summaries
        )
        
        return formatted_document
    
    def _replace_occurrences(self, text: str, target: str, replacements: List[str]) -> str:
        """
        Replace occurrences of a target placeholder with corresponding replacements.

        Args:
            text: Text containing placeholders
            target: Placeholder to replace
            replacements: List of replacements for each occurrence

        Returns:
            Text with replacements
        """
        result = text
        for counter, replacement in enumerate(replacements):
            if target in result:
                if replacement.lower() != 'non-informative':
                    result = result.replace(
                        target,
                        f'picture_counter_{counter}' + ' ' + replacement,
                        1
                    )
                else:
                    result = result.replace(target, '', 1)

        return result

    def chunk_document(self, formatted_document: str) -> List[str]:
        """
        Split the document into semantic chunks.

        Args:
            formatted_document: Formatted document text

        Returns:
            List of document chunks
        """

        # Check document length and split if too large for context
        MAX_TOKENS = 30000  # Reduced to 30k for hierarchical chunking
        estimated_tokens = len(formatted_document.split())  # Rough estimation

        if estimated_tokens > MAX_TOKENS:
            self.logger.warning(f"Document too large ({estimated_tokens} tokens), using hierarchical chunking")
            return self._hierarchical_chunking(formatted_document)

        # Split by section boundaries
        SPLIT_PATTERN = "\n#"
        chunks = formatted_document.split(SPLIT_PATTERN)

        chunked_text = ""
        for i, chunk in enumerate(chunks):
            if chunk.startswith("#"):
                chunk = f"#{chunk}"  # add the # back to the chunk
            chunked_text += f"<|start_chunk_{i}|>\n{chunk}\n<|end_chunk_{i}|>\n"

        # LLM-based semantic chunking
        CHUNKING_PROMPT = """
        You are an assistant specialized in splitting text into semantically consistent sections.

        Following is the document text:
        <document>
        {document_text}
        </document>

        <instructions>
        Instructions:
            1. The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number.
            2. Identify points where splits should occur, such that consecutive chunks of similar themes stay together.
            3. Each chunk must be between 256 and 512 tokens (approximately 170-340 words).
            4. If chunks 1 and 2 belong together but chunk 3 starts a new topic, suggest a split after chunk 2.
            5. The chunks must be listed in ascending order.
            6. Provide your response in the form: 'split_after: 3, 5'.
        </instructions>

        Respond only with the IDs of the chunks where you believe a split should occur.
        YOU MUST RESPOND WITH AT LEAST ONE SPLIT.
        """.strip()

        formatted_chunking_prompt = CHUNKING_PROMPT.format(document_text=chunked_text)
        chunking_response = self.chunker_model.invoke(formatted_chunking_prompt).content

        return self._split_text_by_llm_suggestions(chunked_text, chunking_response)
    
    def _split_text_by_llm_suggestions(self, chunked_text: str, llm_response: str) -> List[str]:
        """
        Split text according to LLM suggested split points.

        Args:
            chunked_text: Text with chunk markers
            llm_response: LLM response with split suggestions

        Returns:
            List of document chunks
        """
        # Extract split points from LLM response
        split_after = []
        if "split_after:" in llm_response:
            split_points = llm_response.split("split_after:")[1].strip()
            # Clean up the response to handle edge cases like "512." or "3,5"
            import re
            numbers = re.findall(r'\d+', split_points)
            split_after = [int(x) for x in numbers if x.isdigit()]

        # Find all chunk markers in the text
        chunk_pattern = r"<\|start_chunk_(\d+)\|>(.*?)<\|end_chunk_\1\|>"
        chunks = re.findall(chunk_pattern, chunked_text, re.DOTALL)

        # Group chunks according to split points
        sections = []
        current_section = []

        for chunk_id, chunk_text in chunks:
            current_section.append(chunk_text)
            if int(chunk_id) in split_after:
                sections.append("".join(current_section).strip())
                current_section = []

        # Add the last section if it's not empty
        if current_section:
            sections.append("".join(current_section).strip())

        return sections
    

    def _hierarchical_chunking(self, formatted_document: str) -> List[str]:
        """
        Hierarchical chunking method for large documents that exceed context limits.
        Step 1: Split into coarse segments (30k tokens each)
        Step 2: Apply semantic chunking to each segment
        Step 3: Merge results

        Args:
            formatted_document: Formatted document text

        Returns:
            List of document chunks
        """
        self.logger.info("Using hierarchical chunking for large document")

        # Step 1: Split into coarse segments (smaller segments for better chunking)
        coarse_segments = self._split_into_coarse_segments(formatted_document, max_tokens=8000)

        self.logger.info(f"Split document into {len(coarse_segments)} coarse segments")

        all_chunks = []

        # Step 2: Apply semantic chunking to each segment
        for i, segment in enumerate(coarse_segments):
            self.logger.info(f"Processing segment {i+1}/{len(coarse_segments)}")

            # Apply semantic chunking to this segment
            segment_chunks = self._semantic_chunk_segment(segment)

            # Add segment identifier to each chunk
            for chunk in segment_chunks:
                chunk_with_metadata = f"[Segment {i+1}]\n{chunk}"
                all_chunks.append(chunk_with_metadata)

        self.logger.info(f"Hierarchical chunking created {len(all_chunks)} chunks total")
        return all_chunks

    def _split_into_coarse_segments(self, text: str, max_tokens: int = 8000) -> List[str]:
        """
        Split text into coarse segments of approximately max_tokens each.

        Args:
            text: Text to split
            max_tokens: Maximum tokens per segment

        Returns:
            List of text segments
        """
        words = text.split()
        segments = []
        current_segment = []

        for word in words:
            current_segment.append(word)

            # Check if we've reached the token limit (rough estimation)
            if len(current_segment) >= max_tokens:
                segments.append(" ".join(current_segment))
                current_segment = []

        # Add the last segment if it has content
        if current_segment:
            segments.append(" ".join(current_segment))

        return segments

    def _semantic_chunk_segment(self, segment_text: str) -> List[str]:
        """
        Apply semantic chunking to a segment of text.

        Args:
            segment_text: Text segment to chunk

        Returns:
            List of semantic chunks
        """
        # Check if segment is still too large for LLM
        segment_tokens = len(segment_text.split())
        if segment_tokens <= 15000:  # Reduced threshold
            # If segment is small enough, apply normal semantic chunking
            return self._apply_semantic_chunking(segment_text)
        else:
            # If segment is still too large, recursively split it
            sub_segments = self._split_into_coarse_segments(segment_text, max_tokens=6000)  # Smaller segments
            all_chunks = []

            for sub_segment in sub_segments:
                sub_chunks = self._apply_semantic_chunking(sub_segment)
                all_chunks.extend(sub_chunks)

            return all_chunks

    def _apply_semantic_chunking(self, text: str) -> List[str]:
        """
        Apply semantic chunking using LLM on text that fits within context window.

        Args:
            text: Text to chunk semantically

        Returns:
            List of semantic chunks
        """
        # Split by section boundaries
        SPLIT_PATTERN = "\n#"
        chunks = text.split(SPLIT_PATTERN)

        chunked_text = ""
        for i, chunk in enumerate(chunks):
            if chunk.startswith("#"):
                chunk = f"#{chunk}"  # add the # back to the chunk
            chunked_text += f"<|start_chunk_{i}|>\n{chunk}\n<|end_chunk_{i}|>\n"

        # Use same chunking prompt as before
        CHUNKING_PROMPT = """
        You are an assistant specialized in splitting text into semantically consistent sections.

        Following is the document text:
        <document>
        {document_text}
        </document>

        <instructions>
        Instructions:
            1. The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number.
            2. Identify points where splits should occur, such that consecutive chunks of similar themes stay together.
            3. Each chunk must be between 256 and 512 tokens (approximately 170-340 words).
            4. If chunks 1 and 2 belong together but chunk 3 starts a new topic, suggest a split after chunk 2.
            5. The chunks must be listed in ascending order.
            6. Provide your response in the form: 'split_after: 3, 5'.
        </instructions>

        Respond only with the IDs of the chunks where you believe a split should occur.
        YOU MUST RESPOND WITH AT LEAST ONE SPLIT.
        """.strip()

        formatted_chunking_prompt = CHUNKING_PROMPT.format(document_text=chunked_text)
        chunking_response = self.chunker_model.invoke(formatted_chunking_prompt).content

        return self._split_text_by_llm_suggestions(chunked_text, chunking_response)
