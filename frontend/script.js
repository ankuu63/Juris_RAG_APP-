
let selectedPDF = null;

let deleteTargetPDF = null;



// =========================
// LOAD PDFs
// =========================

async function loadPDFs() {

    const pdfList =
        document.getElementById("pdfList");

    try {

        const response =
            await fetch("/pdfs");

        const data =
            await response.json();

        pdfList.innerHTML = "";

        if (data.pdfs.length === 0) {

            pdfList.innerHTML =
                "<p>No PDFs uploaded.</p>";

            return;
        }

        data.pdfs.forEach(pdf => {

            const pdfItem =
                document.createElement("div");

            pdfItem.className =
                "pdf-item";

            if (pdf === selectedPDF) {

                pdfItem.classList.add("active-pdf");
            }

            pdfItem.innerHTML = `

                <div class="pdf-left">

                    <div class="pdf-icon">
                        📄
                    </div>

                    <div class="pdf-name">
                        ${pdf}
                    </div>

                </div>

                <div class="pdf-actions">

                    <button
                        class="chat-btn"
                        data-pdf="${pdf}"
                    >
                        Chat
                    </button>

                    <button
                        class="delete-btn"
                        data-pdf="${pdf}"
                    >
                        Delete
                    </button>

                </div>
            `;

            pdfList.appendChild(pdfItem);

        });

    } catch (error) {

        console.error(error);

        pdfList.innerHTML =
            "Failed to load PDFs";
    }
}



// =========================
// SELECT PDF
// =========================

function selectPDF(pdf) {

    selectedPDF = pdf;

    document
        .querySelectorAll(".pdf-item")
        .forEach(item => {

        item.classList.remove("active-pdf");
    });

    const allPDFNames =
        document.querySelectorAll(".pdf-name");

    allPDFNames.forEach(name => {

        if (name.innerText === pdf) {

            name.closest(".pdf-item")
                .classList.add("active-pdf");
        }
    });

    loadChatHistory(pdf);
}



// =========================
// LOAD CHAT HISTORY
// =========================

async function loadChatHistory(pdf) {

    const chatArea =
        document.getElementById("chatArea");

    chatArea.innerHTML = "";


    // =========================
    // HEADER MESSAGE
    // =========================

    const welcome =
        document.createElement("div");

    welcome.className =
        "bot-message";

    welcome.textContent =
        `Chatting with ${pdf}`;

    chatArea.appendChild(welcome);


    try {

        const response =
            await fetch(`/chat-history/${pdf}`);

        const data =
            await response.json();


        // =========================
        // RENDER STORED MESSAGES
        // =========================

        data.messages.forEach(msg => {

            const messageDiv =
                document.createElement("div");


            // USER MESSAGE

            if (msg.sender === "user") {

                messageDiv.className =
                    "user-message";

                messageDiv.textContent =
                    msg.message;
            }


            // ASSISTANT MESSAGE

            else {

                messageDiv.className =
                    "bot-message";

                messageDiv.innerHTML =
                    `
                        <div class="answer-text">

                            ${msg.message}

                        </div>
                    `;
            }

            chatArea.appendChild(messageDiv);
        });


        // =========================
        // AUTO SCROLL
        // =========================

        chatArea.scrollTop =
            chatArea.scrollHeight;

    }

    catch (error) {

        console.error(error);

        alert("Failed to load chat history");
    }
}



// =========================
// DELETE MODAL
// =========================

function deletePDF(filename) {

    deleteTargetPDF = filename;

    document.getElementById("deleteMessage")
        .textContent =
        `Are you sure you want to delete "${filename}" ?`;

    document.getElementById("deleteModal")
        .style.display = "flex";
}


function closeDeleteModal() {

    document.getElementById("deleteModal")
        .style.display = "none";

    deleteTargetPDF = null;
}


async function confirmDeletePDF() {

    if (!deleteTargetPDF) {
        return;
    }

    try {

        const response =
            await fetch(`/delete-pdf/${deleteTargetPDF}`, {
                method: "DELETE"
            });

        const data =
            await response.json();

        if (response.ok) {

            if (selectedPDF === deleteTargetPDF) {

                selectedPDF = null;

                document.getElementById("chatArea")
                    .innerHTML = `
                        <div class="bot-message">
                            Workspace deleted.
                        </div>
                    `;
            }

            closeDeleteModal();

            loadPDFs();

        } else {

            alert(data.error);
        }

    } catch (error) {

        console.error(error);

        alert("Delete failed");
    }
}



// =========================
// CHAT FORM
// =========================

const form =
    document.getElementById("chatForm");

form.addEventListener(
    "submit",
    async function (e) {

    e.preventDefault();

    const askButton =
        document.getElementById("askButton");

    askButton.disabled = true;

    askButton.innerText = "Processing...";

    const chatArea =
        document.getElementById("chatArea");

    const query =
        document.getElementById("query").value;

    const formData =
        new FormData();

    formData.append("query", query);


    // EXISTING PDF

    if (selectedPDF) {

        formData.append(
            "selected_pdf",
            selectedPDF
        );
    }


    // NEW PDF WORKSPACE

    const pdfFile =
        document.getElementById("pdf").files[0];

    if (pdfFile) {

        formData.append("pdf", pdfFile);

        selectedPDF = pdfFile.name;

        // CLEAR OLD CHAT

        chatArea.innerHTML = "";

        // SHOW HEADER

        const welcome =
            document.createElement("div");

        welcome.className =
            "bot-message";

        welcome.textContent =
            `Chatting with ${selectedPDF}`;

        chatArea.appendChild(welcome);

        // REFRESH SIDEBAR FIRST

        await loadPDFs();
    }


    // USER MESSAGE

    const userMessage =
        document.createElement("div");

    userMessage.className =
        "user-message";

    userMessage.textContent =
        query;

    chatArea.appendChild(userMessage);


    try {

        const response =
            await fetch("/chat", {
                method: "POST",
                body: formData
            });

        const data =
            await response.json();

        const botMessage =
            document.createElement("div");

        botMessage.className =
            "bot-message";


        // =========================
        // BUILD SOURCES HTML
        // =========================

        let sourcesHTML = "";

        if (data.sources && data.sources.length > 0) {

            sourcesHTML += `
                <div class="citation-box">

                    <strong>Sources:</strong>

                    <ul>
            `;

            data.sources.forEach(source => {

                sourcesHTML += `
                    <li>
                        📄 ${source.file} — Page ${source.page}
                    </li>
                `;
            });

            sourcesHTML += `
                    </ul>

                </div>
            `;
        }


        // =========================
        // FINAL BOT MESSAGE
        // =========================

        botMessage.innerHTML = `

            <div class="answer-text">

                ${data.answer || data.error}

            </div>

            ${sourcesHTML}
        `;

        chatArea.appendChild(botMessage);


        chatArea.scrollTop =
            chatArea.scrollHeight;

        document.getElementById("query").value = "";

        document.getElementById("pdf").value = "";

    } catch (error) {

        alert("Network Error");
    }

    finally {

        askButton.disabled = false;

        askButton.innerText = "Ask";
    }

});



// =========================
// GLOBAL BUTTON EVENTS
// =========================

document.addEventListener("click", function (e) {

    // CHAT BUTTON

    if (e.target.classList.contains("chat-btn")) {

        const pdf =
            e.target.dataset.pdf;

        selectPDF(pdf);
    }


    // DELETE BUTTON

    if (e.target.classList.contains("delete-btn")) {

        const pdf =
            e.target.dataset.pdf;

        deletePDF(pdf);
    }

});



// INITIAL LOAD

loadPDFs();
