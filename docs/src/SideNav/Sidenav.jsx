import styles from "./Sidenav.module.css"
function SideNav({title="Menu",pages=[{link:"#home",text:"Home"}]}){
    const listPages = pages.map(page => <li key={page.text}><a href={page.link}>{page.text}</a></li>)
    return(
    <div className={styles.nav}>
        <h2>{title}</h2>
        <ul>
            {listPages}
        </ul>
    </div>
    );
}

export default SideNav