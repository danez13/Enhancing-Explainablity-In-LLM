import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import SideNav from "./SideNav/Sidenav.jsx"
import Topbar from "./Topbar/Topbar.jsx"
import Contentbody from "./homePage/homePage.jsx"
import "./docs.css"
function Docs(){
    return (
        <>
            <Topbar />
            <hr />
            <br />
            <div className="content">
                <SideNav
                    title="Table of Contents"
                    pages={[
                    {link:"#home",text:"Home"},
                    {link:"#introduction",text:"Introduction"}
                    ]}/>
                <Contentbody/>
            </div>
        </>
    );
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Docs />
  </StrictMode>,
)
