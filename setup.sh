mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"supravat_ray_ampba2022s@isb.edu\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml