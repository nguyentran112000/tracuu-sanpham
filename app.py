import os
import uuid
import math
import streamlit as st
import chromadb
from PIL import Image
from sentence_transformers import SentenceTransformer
import easyocr
import numpy as np
from streamlit_paste_button import paste_image_button

# Tắt các cảnh báo không cần thiết từ Hugging Face
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Cấu hình trang
st.set_page_config(
    page_title="Hệ thống Nhận diện & Tra cứu Sản phẩm (AI + OCR)",
    page_icon="🔍",
    layout="wide"
)

IMAGE_STORAGE_DIR = "./registered_images"
os.makedirs(IMAGE_STORAGE_DIR, exist_ok=True)

@st.cache_resource
def load_clip_model():
    return SentenceTransformer('clip-ViT-B-32')

@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['vi', 'en'], gpu=False)

@st.cache_resource
def init_vector_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(
        name="food_products",
        metadata={"hnsw:space": "cosine"}
    )

model = load_clip_model()
reader = load_ocr_reader()
collection = init_vector_db()

def get_image_embedding(image: Image.Image):
    image_rgb = image.convert('RGB')
    return model.encode(image_rgb).tolist()

def extract_text_from_image(image: Image.Image):
    img_np = np.array(image.convert('RGB'))
    results = reader.readtext(img_np, detail=0)
    extracted_text = " ".join(results).upper()
    return extracted_text

# Giao diện chính
st.title("🔍 Hệ Thống Tra Cứu Sản Phẩm (AI Hình Ảnh + Đọc Chữ OCR)")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔎 Tra Cứu Hình Ảnh", "➕ Đăng Ký Sản Phẩm Mới", "📋 Danh Sách Sản Phẩm Đã Lưu"])

# TAB 1: TRA CỨU SẢN PHẨM
with tab1:
    st.header("Tra Cứu Nhanh Sản Phẩm")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Tải lên, Chụp ảnh hoặc Dán ảnh sản phẩm")
        source_option = st.radio(
            "Nguồn ảnh:", 
            ["Tải file lên (Upload)", "Dán ảnh từ Zalo / Clipboard", "Chụp từ Camera"], 
            horizontal=True
        )
        
        query_image = None
        
        if source_option == "Tải file lên (Upload)":
            uploaded_file = st.file_uploader("Kéo thả hoặc chọn file ảnh sản phẩm:", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file:
                query_image = Image.open(uploaded_file)
                
        elif source_option == "Dán ảnh từ Zalo / Clipboard":
            st.write("Sao chép (Copy) ảnh từ Zalo hoặc Màn hình, sau đó nhấn nút bên dưới:")
            paste_result = paste_image_button(
                label="📋 Click vào đây để Dán ảnh đã Copy",
                background_color="#1F77B4",
                text_color="#FFFFFF",
                hover_background_color="#145A86"
            )
            if paste_result.image_data is not None:
                query_image = paste_result.image_data
                
        else:
            camera_file = st.camera_input("Chụp ảnh trực tiếp sản phẩm:")
            if camera_file:
                query_image = Image.open(camera_file)
        
        if query_image:
            st.image(query_image, caption="Ảnh bạn đã chọn/dán", use_container_width=True)
            threshold = st.slider("Ngưỡng tin cậy hình ảnh (%)", min_value=50, max_value=95, value=70, step=5) / 100.0
            search_btn = st.button("🚀 Bắt đầu Tra Cứu", type="primary", use_container_width=True)

    with col2:
        st.subheader("2. Kết quả nhận diện (AI + Trích xuất văn bản)")
        if query_image and 'search_btn' in locals() and search_btn:
            with st.spinner("Đang phân tích hình ảnh & đọc chữ trên bao bì..."):
                detected_text = extract_text_from_image(query_image)
                query_vector = get_image_embedding(query_image)
                results = collection.query(query_embeddings=[query_vector], n_results=5)
                
                if not results['ids'] or not results['ids'][0]:
                    st.warning("⚠️ Cơ sở dữ liệu chưa có sản phẩm nào. Vui lòng sang Tab 'Đăng Ký Sản Phẩm Mới' để nhập dữ liệu mẫu.")
                else:
                    best_match = None
                    highest_score = -1.0
                    
                    for idx in range(len(results['ids'][0])):
                        doc_id = results['ids'][0][idx]
                        metadata = results['metadatas'][0][idx]
                        distance = results['distances'][0][idx]
                        similarity = 1.0 - distance
                        
                        p_id = str(metadata.get('product_id', '')).upper()
                        p_name = str(metadata.get('product_name', '')).upper()
                        
                        text_matched = False
                        if (p_id and p_id in detected_text) or (p_name and p_name in detected_text):
                            text_matched = True
                        
                        final_score = similarity + (0.3 if text_matched else 0.0)
                        
                        if final_score > highest_score:
                            highest_score = final_score
                            best_match = {
                                "doc_id": doc_id,
                                "metadata": metadata,
                                "similarity": similarity,
                                "text_matched": text_matched
                            }
                    
                    if best_match and (best_match['similarity'] >= threshold or best_match['text_matched']):
                        meta = best_match['metadata']
                        st.success(f"✅ **ĐÃ XÁC NHẬN CHÍNH XÁC SẢN PHẨM**")
                        if best_match['text_matched']:
                            st.info("💡 **Hệ thống đã khớp thành công chữ in/mã hàng trên thùng carton!**")
                        
                        st.markdown(f"""
                        * **Mã Sản Phẩm:** `{meta.get('product_id', best_match['doc_id'])}`
                        * **Tên Sản Phẩm:** **{meta.get('product_name', 'Không có')}**
                        * **Khách Hàng:** **{meta.get('customer_name', 'Không có')}**
                        * **Dạng Bao Bì:** **{meta.get('package_type', 'Không có')}**
                        * **Tiêu Chuẩn Đóng Gói:** {meta.get('packaging_spec', 'Không có')}
                        """)
                        
                        st.markdown(f"**Chữ AI tự đọc được trên bao bì:** `{detected_text}`")
                        
                        registered_img_path = meta.get('image_path')
                        if registered_img_path and os.path.exists(registered_img_path):
                            st.image(registered_img_path, caption=f"Ảnh mẫu trong hệ thống ({meta.get('product_name', '')})", use_container_width=True)
                    else:
                        st.error(f"❌ **KHÔNG TÌM THẤY SẢN PHẨM PHÙ HỢP IN DATABASE**")
                        st.write(f"Chữ AI đọc được từ ảnh của bạn: `{detected_text}`")

# TAB 2: ĐĂNG KÝ SẢN PHẨM MỚI
with tab2:
    st.header("Đăng Ký Sản Phẩm Mẫu Về Hệ Thống")
    with st.form("add_product_form", clear_on_submit=True):
        col_a, col_b = st.columns(2)
        with col_a:
            product_id = st.text_input("Mã sản phẩm / Item Code *", placeholder="VD: 1748, 1749, T1851, SP001...")
            product_name = st.text_input("Tên sản phẩm *", placeholder="VD: BÁNH TÉT CHUỐI, BÁNH TÉT ĐẬU, BÁNH CHƯNG...")
            customer_name = st.text_input("Khách hàng *", placeholder="VD: LB, Duy Khoi Foods, AFOODS...")
            package_type = st.text_input("Dạng bao bì *", placeholder="VD: Thùng, Túi PE 500g, Khay PA...")
            packaging_spec = st.text_area("Tiêu chuẩn đóng gói *", placeholder="VD: Đóng gói 6 kg/thùng, mạ băng 10%...")
        with col_b:
            sample_image_file = st.file_uploader("Tải lên ảnh mẫu sản phẩm *", type=["jpg", "jpeg", "png", "webp"])
            if sample_image_file:
                st.image(Image.open(sample_image_file), caption="Ảnh mẫu sản phẩm", use_container_width=True)

        submitted = st.form_submit_button("💾 Lưu Sản Phẩm Vào Data", type="primary", use_container_width=True)
        if submitted:
            if not product_id or not product_name or not customer_name or not package_type or not packaging_spec or not sample_image_file:
                st.error("Vui lòng điền đầy đủ tất cả thông tin có dấu (*) và chọn ảnh mẫu!")
            else:
                try:
                    db_id = product_id.strip()
                    file_ext = os.path.splitext(sample_image_file.name)[1]
                    saved_img_filename = f"{db_id}_{customer_name}{file_ext}".replace(" ", "_").replace("/", "_")
                    saved_img_path = os.path.join(IMAGE_STORAGE_DIR, saved_img_filename)
                    
                    sample_img = Image.open(sample_image_file)
                    sample_img.save(saved_img_path)
                    
                    vector = get_image_embedding(sample_img)
                    collection.add(
                        ids=[db_id],
                        embeddings=[vector],
                        metadatas=[{
                            "product_id": db_id,
                            "customer_name": customer_name,
                            "product_name": product_name,
                            "package_type": package_type,
                            "packaging_spec": packaging_spec,
                            "image_path": saved_img_path
                        }]
                    )
                    st.success(f"🎉 Đã nạp thành công sản phẩm **{product_name}** (Mã: `{db_id}`) của **{customer_name}** vào cơ sở dữ liệu!")
                except Exception as e:
                    st.error(f"Lỗi khi lưu dữ liệu (Mã sản phẩm có thể đã tồn tại): {e}")

# TAB 3: DANH SÁCH, TÌM KIẾM, PHÂN TRANG, CHỈNH SỬA & XÓA SẢN PHẨM
with tab3:
    st.header("Danh Sách Sản Phẩm Trong Cơ Sở Dữ Liệu")
    all_data = collection.get()
    
    if not all_data['ids']:
        st.info("Chưa có sản phẩm nào được đăng ký.")
    else:
        # Thanh tìm kiếm và bộ lọc
        search_kw = st.text_input("🔍 Tìm kiếm nhanh theo Tên, Mã sản phẩm hoặc Khách hàng:", placeholder="Nhập từ khóa cần tìm...").strip().lower()
        
        # Lọc danh sách theo từ khóa tìm kiếm
        filtered_items = []
        for idx, item_id in enumerate(all_data['ids']):
            metadata = all_data['metadatas'][idx]
            p_id = str(metadata.get('product_id', item_id)).lower()
            p_name = str(metadata.get('product_name', '')).lower()
            c_name = str(metadata.get('customer_name', '')).lower()
            
            if not search_kw or (search_kw in p_id or search_kw in p_name or search_kw in c_name):
                filtered_items.append((item_id, metadata))
                
        total_items = len(filtered_items)
        st.write(f"Tổng số sản phẩm phù hợp: **{total_items}** / {len(all_data['ids'])} sản phẩm")
        
        if total_items == 0:
            st.warning("Không tìm thấy sản phẩm nào khớp với từ khóa tìm kiếm.")
        else:
            # Cấu hình phân trang (12 sản phẩm trên 1 trang)
            ITEMS_PER_PAGE = 12
            total_pages = math.ceil(total_items / ITEMS_PER_PAGE)
            
            page_col1, page_col2 = st.columns([1, 4])
            with page_col1:
                current_page = st.number_input("Trang số:", min_value=1, max_value=max(1, total_pages), value=1, step=1)
            with page_col2:
                st.write(f"<br>Hiển thị Trang **{current_page}** / **{total_pages}**", unsafe_allow_html=True)
                
            start_idx = (current_page - 1) * ITEMS_PER_PAGE
            end_idx = start_idx + ITEMS_PER_PAGE
            page_items = filtered_items[start_idx:end_idx]

            cols = st.columns(2)
            if "edit_id" not in st.session_state:
                st.session_state.edit_id = None

            for idx, (item_id, metadata) in enumerate(page_items):
                with cols[idx % 2]:
                    with st.container(border=True):
                        img_path = metadata.get('image_path')
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, use_container_width=True)
                        st.subheader(metadata.get('product_name', 'Sản phẩm không tên'))
                        st.caption(f"🆔 **Mã SP:** `{metadata.get('product_id', item_id)}`")
                        st.write(f"🏢 **Khách hàng:** {metadata.get('customer_name', 'Không có')}")
                        st.write(f"📦 **Dạng bao bì:** {metadata.get('package_type', 'Không có')}")
                        st.write(f"📋 **Tiêu chuẩn đóng gói:** {metadata.get('packaging_spec', 'Không có')}")
                        
                        btn_col1, btn_col2 = st.columns(2)
                        with btn_col1:
                            if st.button(f"✏️ Chỉnh sửa", key=f"edit_btn_{item_id}", use_container_width=True):
                                st.session_state.edit_id = item_id if st.session_state.edit_id != item_id else None
                                st.rerun()
                        with btn_col2:
                            if st.button(f"🗑️ Xóa sản phẩm", key=f"del_btn_{item_id}", type="secondary", use_container_width=True):
                                collection.delete(ids=[item_id])
                                if img_path and os.path.exists(img_path):
                                    try:
                                        os.remove(img_path)
                                    except Exception:
                                        pass
                                st.success("Đã xóa sản phẩm thành công!")
                                st.rerun()

                        # Khung cập nhật thông tin
                        if st.session_state.edit_id == item_id:
                            st.markdown("---")
                            st.write("##### 📝 Cập nhật thông tin sản phẩm:")
                            with st.form(key=f"edit_form_{item_id}"):
                                new_prod_id = st.text_input("Mã sản phẩm", value=metadata.get('product_id', item_id))
                                new_prod_name = st.text_input("Tên sản phẩm", value=metadata.get('product_name', ''))
                                new_cust_name = st.text_input("Khách hàng", value=metadata.get('customer_name', ''))
                                new_pkg_type = st.text_input("Dạng bao bì", value=metadata.get('package_type', ''))
                                new_pkg_spec = st.text_area("Tiêu chuẩn đóng gói", value=metadata.get('packaging_spec', ''))
                                new_img_file = st.file_uploader("Thay ảnh mẫu mới", type=["jpg", "jpeg", "png", "webp"])
                                
                                c1, c2 = st.columns(2)
                                with c1:
                                    save_edit = st.form_submit_button("💾 Lưu thay đổi", type="primary", use_container_width=True)
                                with c2:
                                    cancel_edit = st.form_submit_button("❌ Hủy", use_container_width=True)

                                if save_edit:
                                    current_img_path = metadata.get('image_path')
                                    vector = None
                                    
                                    if new_img_file:
                                        sample_img = Image.open(new_img_file)
                                        vector = get_image_embedding(sample_img)
                                        
                                        file_ext = os.path.splitext(new_img_file.name)[1]
                                        saved_img_filename = f"{new_prod_id}_{new_cust_name}{file_ext}".replace(" ", "_").replace("/", "_")
                                        current_img_path = os.path.join(IMAGE_STORAGE_DIR, saved_img_filename)
                                        sample_img.save(current_img_path)

                                    updated_metadata = {
                                        "product_id": new_prod_id,
                                        "customer_name": new_cust_name,
                                        "product_name": new_prod_name,
                                        "package_type": new_pkg_type,
                                        "packaging_spec": new_pkg_spec,
                                        "image_path": current_img_path
                                    }

                                    if vector:
                                        collection.update(ids=[item_id], embeddings=[vector], metadatas=[updated_metadata])
                                    else:
                                        collection.update(ids=[item_id], metadatas=[updated_metadata])

                                    st.session_state.edit_id = None
                                    st.success("Cập nhật thông tin thành công!")
                                    st.rerun()

                                if cancel_edit:
                                    st.session_state.edit_id = None
                                    st.rerun()
