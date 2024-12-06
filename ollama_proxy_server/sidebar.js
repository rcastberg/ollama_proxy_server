// Function to dynamically load and inject the sidebar HTML
window.onload = function() {
    console.log("Loading sidebar...");
    const sidebarHTML = `
      <div id="sidebar">
        <ul>
          <li><a href="/local/view_stats">View Stats</a></li>
          <li><a href="/local/user_admin">User Admin</a></li>
          <li><a href="/local/server_admin">Server Admin</a></li>
        </ul>
      </div>
      <style>
        #sidebar {
          width: 250px;
          background-color: #333;
          color: white;
          padding-top: 20px;
          height: 100%; /* Full height */
          position: fixed;
          top: 0;
          left: 0;
          flex-shrink: 0; /* Prevent sidebar from shrinking */
        }

        #sidebar ul {
          list-style-type: none;
          padding: 0;
        }

        #sidebar ul li {
          padding: 10px;
          text-align: center;
        }

        #sidebar ul li a {
          color: white;
          text-decoration: none;
          font-size: 18px;
        }

        #sidebar ul li a:hover {
          background-color: #575757;
          display: block;
          padding: 10px;
        }

        /* Wrapper to hold both sidebar and content */
        #wrapper {
          display: flex; /* Flexbox layout to position side by side */
          min-height: 100vh;
        }

        /* Content section */
        #content {
          margin-left: 250px; /* Ensure content starts after the sidebar */
          padding: 20px;
          flex-grow: 1; /* Take up remaining space */
        }
      </style>
    `;

    // Insert the sidebar HTML into the document
    document.body.insertAdjacentHTML('afterbegin', sidebarHTML);
  };
